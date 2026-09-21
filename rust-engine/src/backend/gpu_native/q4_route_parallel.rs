//! Production Q4 route scheduling replicated from the hardware-qualified PR2-C.
//! The frozen serial control encoder stays unchanged in the parent.
use super::*;

// Compile the exact qualified shader sources. Historical serial entrypoints
// and arithmetic pins stay intact for the frozen control.
const ROUTE_PARALLEL_SHADER: &str = concat!(
    include_str!("../wgpu_shaders/gpu_native_q4_expert.wgsl"),
    include_str!("../wgpu_shaders/gpu_native_q4_expert_route_parallel.wgsl"),
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RouteParallelGeometry {
    pub(crate) gate_up_workgroups: u32,
    pub(crate) down_workgroups: u32,
    pub(crate) route_width: u32,
    pub(crate) activation_elements: usize,
}
impl RouteParallelGeometry {
    pub(crate) fn try_new(
        g: GpuNativeQ4ExpertGeometry,
        limits: &wgpu::Limits,
    ) -> Result<Self, GpuNativeBootstrapError> {
        let overflow = || GpuNativeBootstrapError::ExpertGeometryOverflow {
            d_model: g.d_model,
            d_ff: g.d_ff,
            num_experts: g.num_experts,
            top_k: g.top_k,
        };
        let activation_elements = g.top_k.checked_mul(g.d_ff).ok_or_else(overflow)?;
        u32::try_from(activation_elements).map_err(|_| overflow())?;
        let result = Self {
            gate_up_workgroups: u32::try_from(g.d_ff.div_ceil(64)).map_err(|_| overflow())?,
            down_workgroups: u32::try_from(g.d_model.div_ceil(64)).map_err(|_| overflow())?,
            route_width: u32::try_from(g.top_k).map_err(|_| overflow())?,
            activation_elements,
        };
        for workgroups in [
            result.gate_up_workgroups,
            result.down_workgroups,
            result.route_width,
        ] {
            if workgroups > limits.max_compute_workgroups_per_dimension {
                return Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
                    workgroups: u64::from(workgroups),
                    maximum: limits.max_compute_workgroups_per_dimension,
                });
            }
        }
        let layout = GpuNativeScratchLayout::try_new(activation_elements)?;
        super::super::validate_startup_buffer(
            "pr2c_activation",
            layout.bytes,
            wgpu::BufferUsages::STORAGE,
            limits,
        )?;
        Ok(result)
    }
}

/// Production request storage-only top_k*d_ff activation and its two pipelines.
/// Locations, outputs, status and combined remain shared scratch.
pub(crate) struct GpuNativeQ4RouteParallelScratch {
    context_id: u64,
    geometry: GpuNativeQ4ExpertGeometry,
    activation: GpuNativeScratch,
}

/// Executor-owned route-parallel pipelines. These are invariant across
/// requests and are created once with the other GPU-native pipelines.
pub(super) struct GpuNativeQ4RouteParallelPipelines {
    gate_up: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
}

impl GpuNativeQ4RouteParallelPipelines {
    pub(super) fn create_for_executor(
        device: &wgpu::Device,
        q4_expert_pipelines: &GpuNativeQ4ExpertPipelines,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pr2c_q4_route_parallel_shader"),
            source: wgpu::ShaderSource::Wgsl(ROUTE_PARALLEL_SHADER.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pr2c_q4_route_parallel_layout"),
            bind_group_layouts: &[&q4_expert_pipelines.expert_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..GPU_NATIVE_EXPERT_PUSH_CONSTANT_BYTES,
            }],
        });
        let pipeline = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&layout),
                module: &module,
                entry_point,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        };
        Self {
            gate_up: pipeline("q4_expert_gate_up_route_parallel_main"),
            down: pipeline("q4_expert_down_route_parallel_main"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct DispatchEvidence {
    pub(crate) layer_executions: u64,
    pub(crate) accounting_mismatch: bool,
    pub(crate) selected_routes: u64,
    pub(crate) top_k: u64,
    pub(crate) serial_gate_up_dispatches: u64,
    pub(crate) serial_down_dispatches: u64,
    pub(crate) parallel_gate_up_dispatches: u64,
    pub(crate) parallel_down_dispatches: u64,
    pub(crate) routes_covered: u64,
    pub(crate) max_route_width: u64,
    pub(crate) activation_bytes: u64,
    pub(crate) route_output_bytes: u64,
    pub(crate) gate_up_rows: u64,
    pub(crate) down_rows: u64,
    pub(crate) validation_dispatches: u64,
    pub(crate) combine_dispatches: u64,
    pub(crate) contain_dispatches: u64,
}

// Preserve the qualified bounded counter accounting in the production encoder.
// Only an installed qualification observer aggregates the returned evidence.
#[derive(Clone, Copy)]
struct Q4DispatchCounters {
    resolve: u64,
    gate_up: u64,
    down: u64,
    combine: u64,
}
impl Q4DispatchCounters {
    fn capture(executor: &GpuNativeExecutorContext) -> Self {
        Self {
            resolve: executor
                .counters
                .expert_route_resolve_dispatches
                .load(Ordering::Relaxed),
            gate_up: executor
                .counters
                .q4_expert_gate_up_dispatches
                .load(Ordering::Relaxed),
            down: executor
                .counters
                .q4_expert_down_dispatches
                .load(Ordering::Relaxed),
            combine: executor
                .counters
                .expert_combine_dispatches
                .load(Ordering::Relaxed),
        }
    }
    fn delta(self, before: Self) -> Self {
        let delta = |after: u64, before: u64| after.checked_sub(before).unwrap_or(u64::MAX);
        Self {
            resolve: delta(self.resolve, before.resolve),
            gate_up: delta(self.gate_up, before.gate_up),
            down: delta(self.down, before.down),
            combine: delta(self.combine, before.combine),
        }
    }
}

impl GpuNativeExecutorContext {
    pub(crate) fn create_q4_route_parallel_scratch(
        &self,
        geometry: GpuNativeQ4ExpertGeometry,
    ) -> Result<GpuNativeQ4RouteParallelScratch, GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        let plan = RouteParallelGeometry::try_new(geometry, &gpu.device.limits())?;
        let activation = self.create_scratch(plan.activation_elements)?;
        Ok(GpuNativeQ4RouteParallelScratch {
            context_id: self.context_id,
            geometry,
            activation,
        })
    }

    /// Control observes the counters surrounding the exact frozen pre-C serial
    /// encoder. No synthetic serial implementation.
    pub(crate) fn encode_q4_expert_serial_qualification(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        router_plan: &GpuNativeRouterPlan,
        router_scratch: &GpuNativeRouterScratch,
        arena: &GpuNativeQ4ExpertArena,
        state: &GpuNativeTokenState,
        expert_scratch: &GpuNativeQ4ExpertScratch,
    ) -> Result<DispatchEvidence, GpuNativeBootstrapError> {
        let before = Q4DispatchCounters::capture(self);
        self.encode_q4_expert_arena_combine(
            encoder,
            router_plan,
            router_scratch,
            arena,
            state,
            expert_scratch,
        )?;
        let delta = Q4DispatchCounters::capture(self).delta(before);
        let k = arena.geometry.top_k as u64;
        Ok(DispatchEvidence {
            layer_executions: delta.resolve,
            selected_routes: k,
            top_k: k,
            serial_gate_up_dispatches: delta.gate_up,
            serial_down_dispatches: delta.down,
            activation_bytes: expert_scratch.layout.activation_bytes(),
            route_output_bytes: expert_scratch.layout.route_outputs_bytes(),
            gate_up_rows: k * arena.geometry.d_ff as u64,
            down_rows: k * arena.geometry.d_model as u64,
            validation_dispatches: 1,
            combine_dispatches: delta.combine,
            contain_dispatches: 1,
            ..DispatchEvidence::default()
        })
    }

    /// The single production encoder, also called by qualification treatment.
    pub(crate) fn encode_q4_expert_route_parallel(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        router_plan: &GpuNativeRouterPlan,
        router_scratch: &GpuNativeRouterScratch,
        arena: &GpuNativeQ4ExpertArena,
        state: &GpuNativeTokenState,
        expert_scratch: &GpuNativeQ4ExpertScratch,
        parallel: &GpuNativeQ4RouteParallelScratch,
    ) -> Result<DispatchEvidence, GpuNativeBootstrapError> {
        self.encode_q4_expert_route_parallel_with_binding(
            encoder,
            router_plan,
            router_scratch,
            arena,
            state,
            expert_scratch,
            parallel,
            None,
        )
    }

    pub(crate) fn encode_q4_expert_route_parallel_p1j(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        router_plan: &GpuNativeRouterPlan,
        router_scratch: &GpuNativeRouterScratch,
        arena: &GpuNativeQ4ExpertArena,
        state: &GpuNativeTokenState,
        expert_scratch: &GpuNativeQ4ExpertScratch,
        parallel: &GpuNativeQ4RouteParallelScratch,
        sidecar: &GpuNativeP1jSidecar,
        id: crate::predictor_v2::P1jIdentity,
    ) -> Result<DispatchEvidence, GpuNativeBootstrapError> {
        if sidecar.context_id != self.context_id
            || sidecar.arena_identity != arena as *const _ as usize
            || id.candidate.namespace.arena != sidecar.arena_identity
        {
            return Err(GpuNativeBootstrapError::ForeignExpertArena);
        }
        self.encode_q4_expert_route_parallel_with_binding(
            encoder,
            router_plan,
            router_scratch,
            arena,
            state,
            expert_scratch,
            parallel,
            Some(sidecar),
        )
    }

    fn encode_q4_expert_route_parallel_with_binding(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        router_plan: &GpuNativeRouterPlan,
        router_scratch: &GpuNativeRouterScratch,
        arena: &GpuNativeQ4ExpertArena,
        state: &GpuNativeTokenState,
        expert_scratch: &GpuNativeQ4ExpertScratch,
        parallel: &GpuNativeQ4RouteParallelScratch,
        sidecar: Option<&GpuNativeP1jSidecar>,
    ) -> Result<DispatchEvidence, GpuNativeBootstrapError> {
        let before = Q4DispatchCounters::capture(self);
        let gpu = self.authoritative_gpu()?;
        let limits = gpu.device.limits();
        let geometry = RouteParallelGeometry::try_new(arena.geometry, &limits)?;
        validate_route_parallel_scratch_identity(
            self.context_id,
            arena.geometry,
            parallel.context_id,
            parallel.geometry,
        )?;
        let mut evidence = DispatchEvidence::default();
        validate_token_state_owner(self.context_id, state.context_id)?;
        let _gate = validate_router_plan_with_registry(
            self.context_id,
            self.layout.d_model,
            &self.dense_weights.lock(),
            router_plan,
        )?;
        validate_router_scratch(self.context_id, router_plan.geometry, router_scratch)?;
        validate_q4_expert_arena(self.context_id, arena, &limits)?;
        validate_q4_expert_scratch(self.context_id, arena.geometry, expert_scratch, &limits)?;
        validate_router_expert_geometry(router_plan, arena)?;
        if state.layout.d_model != arena.geometry.d_model {
            return Err(GpuNativeBootstrapError::ExpertDModelMismatch {
                expected: arena.geometry.d_model,
                actual: state.layout.d_model,
            });
        }
        validate_q4_expert_pipeline_limits(&limits)?;
        router_scratch.layout.validate_for_limits(&limits)?;
        state.layout.validate_for_limits(&limits)?;
        let route_workgroups = self.checked_workgroups(arena.geometry.top_k, &limits)?;
        let gate_up_workgroups = self.checked_workgroups(arena.geometry.d_ff, &limits)?;
        let down_workgroups = self.checked_workgroups(arena.geometry.d_model, &limits)?;
        let combine_workgroups = down_workgroups;
        let residual_workgroups = down_workgroups;
        let (active_banks, bank_slots) =
            p1j_binding_layout(arena.plan, arena.layer_index, sidecar.is_some())?;
        let resolve_pc = GpuNativeExpertResolvePushConstants {
            num_experts: arena.geometry.num_experts as u32,
            top_k: arena.geometry.top_k as u32,
            active_banks,
            _reserved: 0,
            bank_slots,
        };
        let combine_pc = GpuNativeExpertCombinePushConstants {
            d_model: arena.geometry.d_model as u32,
            top_k: arena.geometry.top_k as u32,
            _reserved_0: 0,
            _reserved_1: 0,
        };

        // All fallible host-side validation and dispatch planning above is
        // complete before this first expert command is recorded.
        let route_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_expert_route_resolve_bind_group"),
            layout: &self.q4_expert_pipelines.route_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: router_scratch.selected_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: arena.mapping.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: expert_scratch.resolved_locations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.status.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_native_expert_route_resolve_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.q4_expert_pipelines.route_resolve);
            pass.set_bind_group(0, &route_bind_group, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&resolve_pc));
            pass.dispatch_workgroups(route_workgroups, 1, 1);
        }

        let expert_bind_group =
            |label: &'static str, input: &wgpu::Buffer, output: &wgpu::Buffer| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &self.q4_expert_pipelines.expert_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: arena.banks[0].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: sidecar
                                .map_or(&arena.banks[1], |s| &s.buffer)
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: arena.banks[2].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: arena.banks[3].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: input.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: expert_scratch.resolved_locations.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: output.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: state.status.as_entire_binding(),
                        },
                    ],
                })
            };
        let gate_up_bind_group = expert_bind_group(
            "gpu_native_q4_expert_gate_up_bind_group",
            &state.hidden,
            &parallel.activation.buffer,
        );
        let down_bind_group = expert_bind_group(
            "gpu_native_q4_expert_down_bind_group",
            &parallel.activation.buffer,
            &expert_scratch.route_outputs.buffer,
        );
        {
            let pc = GpuNativeQ4ExpertPushConstants {
                d_model: arena.geometry.d_model as u32,
                d_ff: arena.geometry.d_ff as u32,
                blocks_per_projection: arena.geometry.blocks_per_projection as u32,
                slot_stride_bytes: arena.geometry.slot_stride_bytes as u32,
                route_slot: 0,
                top_k: arena.geometry.top_k as u32,
                swiglu_limit: crate::inference::swiglu_limit().unwrap_or(f32::INFINITY),
                _reserved: 0,
            };
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_native_q4_expert_gate_up_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.q4_route_parallel_pipelines.gate_up);
                pass.set_bind_group(0, &gate_up_bind_group, &[]);
                pass.set_push_constants(0, bytemuck::bytes_of(&pc));
                pass.dispatch_workgroups(gate_up_workgroups, geometry.route_width, 1);
                evidence.parallel_gate_up_dispatches += 1;
                evidence.gate_up_rows += (arena.geometry.d_ff * arena.geometry.top_k) as u64;
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_native_q4_expert_down_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.q4_route_parallel_pipelines.down);
                pass.set_bind_group(0, &down_bind_group, &[]);
                pass.set_push_constants(0, bytemuck::bytes_of(&pc));
                pass.dispatch_workgroups(down_workgroups, geometry.route_width, 1);
                evidence.parallel_down_dispatches += 1;
                evidence.down_rows += (arena.geometry.d_model * arena.geometry.top_k) as u64;
                evidence.routes_covered += u64::from(geometry.route_width);
                evidence.max_route_width = u64::from(geometry.route_width);
            }
        }

        let combine_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_expert_combine_bind_group"),
            layout: &self.q4_expert_pipelines.combine_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: router_scratch.selected_weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: expert_scratch.resolved_locations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: expert_scratch.route_outputs.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: expert_scratch.combined.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state.status.as_entire_binding(),
                },
            ],
        });
        let mut control_pass = |label, pipeline: &wgpu::ComputePipeline, workgroups| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            // Control resources intentionally occupy @group(1), but WGPU still
            // requires the pipeline's explicit empty group-0 layout to be bound.
            pass.set_bind_group(0, &self.q4_expert_pipelines.control_empty_bind_group, &[]);
            pass.set_bind_group(1, &combine_bind_group, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&combine_pc));
            pass.dispatch_workgroups(workgroups, 1, 1);
        };
        control_pass(
            "gpu_native_expert_validate_pass",
            &self.q4_expert_pipelines.validate,
            1,
        );
        control_pass(
            "gpu_native_expert_combine_pass",
            &self.q4_expert_pipelines.combine,
            combine_workgroups,
        );
        control_pass(
            "gpu_native_expert_contain_pass",
            &self.q4_expert_pipelines.contain,
            combine_workgroups,
        );
        drop(control_pass);
        evidence.validation_dispatches += 1;
        evidence.combine_dispatches += 1;
        evidence.contain_dispatches += 1;
        self.encode_residual_add_pass(
            gpu,
            encoder,
            state,
            &expert_scratch.combined,
            residual_workgroups,
        );
        self.counters.record_expert_dispatches(1);
        let delta = Q4DispatchCounters::capture(self).delta(before);
        // Fail mechanism accounting closed if the ordinary counters disagree
        // with the dispatches actually recorded at the production seam.
        evidence.accounting_mismatch = delta.gate_up != evidence.parallel_gate_up_dispatches
            || delta.down != evidence.parallel_down_dispatches
            || delta.resolve != 1
            || delta.combine != evidence.combine_dispatches;
        evidence.layer_executions = delta.resolve;
        evidence.selected_routes = arena.geometry.top_k as u64;
        evidence.top_k = arena.geometry.top_k as u64;
        evidence.activation_bytes = parallel.activation.layout.bytes;
        evidence.route_output_bytes = expert_scratch.layout.route_outputs_bytes();
        Ok(evidence)
    }
}

fn validate_route_parallel_scratch_identity(
    context_id: u64,
    geometry: GpuNativeQ4ExpertGeometry,
    scratch_context_id: u64,
    scratch_geometry: GpuNativeQ4ExpertGeometry,
) -> Result<(), GpuNativeBootstrapError> {
    if scratch_context_id != context_id || scratch_geometry != geometry {
        return Err(GpuNativeBootstrapError::ForeignTokenState);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn function(source: &str, name: &str) -> String {
        let start = source.find(&format!("fn {name}(")).unwrap();
        let body = start + source[start..].find('{').unwrap();
        let mut depth = 0;
        for (i, ch) in source[body..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            if depth == 0 {
                return source[start..=body + i].into();
            }
        }
        panic!("unterminated function {name}");
    }
    #[test]
    fn q4_route_parallel_geometry_covers_each_route_row_once_topk_one_through_max() {
        for k in 1..=MAX_GPU_NATIVE_ROUTER_TOP_K {
            for (d_model, d_ff) in [(32, 32), (96, 160), (2048, 768)] {
                let g = GpuNativeQ4ExpertGeometry::try_new(d_model, d_ff, 128, k).unwrap();
                let p = RouteParallelGeometry::try_new(g, &wgpu::Limits::default()).unwrap();
                for (rows, x) in [(d_ff, p.gate_up_workgroups), (d_model, p.down_workgroups)] {
                    let mut visits = vec![0; k * rows];
                    for route in 0..p.route_width as usize {
                        for row in 0..x as usize * 64 {
                            if row < rows {
                                visits[route * rows + row] += 1;
                            }
                        }
                    }
                    assert!(visits.iter().all(|&n| n == 1));
                }
                assert_eq!(p.activation_elements, k * d_ff);
                let production = GpuNativeQ4ExpertScratchLayout::try_new(g).unwrap();
                assert_eq!(production.activation_bytes(), (d_ff * 4) as u64);
                assert_eq!(production.route_outputs_bytes(), (k * d_model * 4) as u64);
            }
        }
    }
    #[test]
    fn q4_route_parallel_y_and_x_dispatch_limits_are_validated_before_encoding() {
        let mut limits = wgpu::Limits::default();
        let g = GpuNativeQ4ExpertGeometry::try_new(32, 32, 128, 8).unwrap();
        limits.max_compute_workgroups_per_dimension = 7;
        assert!(matches!(
            RouteParallelGeometry::try_new(g, &limits),
            Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
                workgroups: 8,
                maximum: 7
            })
        ));
        limits.max_compute_workgroups_per_dimension = 8;
        assert!(RouteParallelGeometry::try_new(g, &limits).is_ok());
        limits.max_compute_workgroups_per_dimension = 0;
        assert!(RouteParallelGeometry::try_new(g, &limits).is_err());
        for (d_model, d_ff) in [(32, 576), (576, 32)] {
            let g = GpuNativeQ4ExpertGeometry::try_new(d_model, d_ff, 128, 8).unwrap();
            limits.max_compute_workgroups_per_dimension = 8;
            assert!(matches!(
                RouteParallelGeometry::try_new(g, &limits),
                Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
                    workgroups: 9,
                    maximum: 8
                })
            ));
        }
    }
    #[test]
    fn q4_route_parallel_foreign_and_mismatched_scratch_fail_closed() {
        let g = GpuNativeQ4ExpertGeometry::try_new(2048, 768, 128, 8).unwrap();
        assert!(validate_route_parallel_scratch_identity(1, g, 1, g).is_ok());
        assert!(matches!(
            validate_route_parallel_scratch_identity(1, g, 2, g),
            Err(GpuNativeBootstrapError::ForeignTokenState)
        ));
        for other in [
            GpuNativeQ4ExpertGeometry::try_new(2048, 768, 128, 1).unwrap(),
            GpuNativeQ4ExpertGeometry::try_new(2048, 800, 128, 8).unwrap(),
        ] {
            assert!(matches!(
                validate_route_parallel_scratch_identity(1, g, 1, other),
                Err(GpuNativeBootstrapError::ForeignTokenState)
            ));
        }
    }
    #[test]
    fn q4_route_parallel_pipelines_are_executor_owned_not_request_local() {
        let source = include_str!("q4_route_parallel.rs");
        let scratch = function(source, "create_q4_route_parallel_scratch");
        let request = function(
            include_str!("../../gpu_native_token_loop.rs"),
            "create_request_state_inner",
        );
        for request_local in [&scratch, &request] {
            for forbidden in [
                "create_shader_module",
                "create_pipeline_layout",
                "create_compute_pipeline",
            ] {
                assert!(!request_local.contains(forbidden), "{forbidden}");
            }
        }
        assert!(scratch.contains("self.create_scratch(plan.activation_elements)?"));

        let pipelines = function(source, "create_for_executor");
        assert_eq!(pipelines.matches("create_shader_module").count(), 1);
        assert_eq!(pipelines.matches("create_pipeline_layout").count(), 1);
        assert_eq!(pipelines.matches("create_compute_pipeline").count(), 1);
        for entry_point in [
            "q4_expert_gate_up_route_parallel_main",
            "q4_expert_down_route_parallel_main",
        ] {
            assert!(pipelines.contains(entry_point));
        }

        let executor = include_str!("../gpu_native.rs");
        let context = executor.find("impl GpuNativeExecutorContext").unwrap();
        let initialize = function(&executor[context..], "try_new");
        let q4 = initialize
            .find("GpuNativeQ4ExpertPipelines::try_new")
            .unwrap();
        let parallel = initialize
            .find("GpuNativeQ4RouteParallelPipelines::create_for_executor")
            .unwrap();
        assert!(q4 < parallel);
        assert_eq!(
            initialize
                .matches("GpuNativeQ4RouteParallelPipelines::create_for_executor")
                .count(),
            1
        );
    }
    #[test]
    fn q4_route_parallel_shader_parses_and_validates_without_a_device() {
        let m = naga::front::wgsl::parse_str(ROUTE_PARALLEL_SHADER).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&m)
        .unwrap();
        for name in [
            "q4_expert_gate_up_route_parallel_main",
            "q4_expert_down_route_parallel_main",
        ] {
            let e = m.entry_points.iter().find(|e| e.name == name).unwrap();
            assert_eq!(e.workgroup_size, [64, 1, 1]);
        }
    }
    #[test]
    fn q4_route_parallel_pins_production_arithmetic_and_swiglu_exactly() {
        use sha2::Digest;
        assert_eq!(
            format!(
                "{:x}",
                sha2::Sha256::digest(GPU_NATIVE_Q4_EXPERT_SHADER.as_bytes())
            ),
            "54bae352aad7f25920212f596c00b8e3bac2c0a14281b9664d219d1eea212306"
        );
        assert_eq!(ROUTE_PARALLEL_SHADER.matches("fn q4_dot(").count(), 1);
        assert_eq!(
            function(ROUTE_PARALLEL_SHADER, "q4_dot"),
            function(GPU_NATIVE_Q4_EXPERT_SHADER, "q4_dot")
        );
        for stage in ["gate_up", "down"] {
            let name = format!("q4_expert_{stage}_main");
            let parallel_name = format!("q4_expert_{stage}_route_parallel_main");
            let actual = function(ROUTE_PARALLEL_SHADER, &parallel_name);
            let expected = function(GPU_NATIVE_Q4_EXPERT_SHADER, &name)
                .replace(&name, &parallel_name)
                .replace(
                    "let row = gid.x;",
                    "let row = gid.x;\n    let route_slot = gid.y;",
                )
                .replace(
                    "if (row >= pc.d_ff)",
                    "if (row >= pc.d_ff || route_slot >= pc.top_k)",
                )
                .replace(
                    "if (row >= pc.d_model)",
                    "if (row >= pc.d_model || route_slot >= pc.top_k)",
                )
                .replace("resolved_route()", "resolved_route_parallel(route_slot)")
                .replace("pc.route_slot", "route_slot")
                .replace("OUTPUT[row]", "OUTPUT[route_slot * pc.d_ff + row]")
                .replace(
                    "first_block, blocks_per_row, 0u)",
                    "first_block, blocks_per_row, route_slot * pc.d_ff)",
                );
            assert_eq!(
                actual, expected,
                "only route addressing and bounds may change"
            );
        }
    }
    #[test]
    fn q4_route_parallel_keeps_control_path_and_gpu_boundary_contract() {
        let runtime = include_str!("../../gpu_native_token_loop.rs");
        let encoded = function(runtime, "encode_expert_layer_unified");
        assert!(encoded.contains("if q4_uses_frozen_serial_control(observation.map(|o| o.arm))"));
        assert!(encoded.contains("encode_q4_expert_serial_qualification"));
        assert_eq!(
            encoded.matches(".encode_q4_expert_route_parallel(").count(),
            1
        );
        assert!(!encoded.contains("encode_q4_expert_arena_combine"));
        assert!(!encoded.contains("Q4QualificationArm::Treatment"));
        let require = encoded.find("require_q4_route_parallel_scratch(").unwrap();
        assert!(require < encoded.find(".encode_q4_expert_route_parallel(").unwrap());
        let allocation = function(runtime, "create_request_state_inner");
        assert!(allocation.contains("if q4_uses_frozen_serial_control("));
        assert!(allocation.contains("create_q4_route_parallel_scratch(expert_geom)?"));
        assert!(!allocation.contains("Q4QualificationArm::Treatment"));
        let source = include_str!("q4_route_parallel.rs");
        assert!(function(source, "encode_q4_expert_serial_qualification")
            .contains("self.encode_q4_expert_arena_combine("));
        let treatment = function(source, "encode_q4_expert_route_parallel_with_binding");
        let ordinary = function(source, "encode_q4_expert_route_parallel")
            .split_whitespace()
            .collect::<String>();
        assert!(ordinary.contains("parallel,None,"));
        assert!(function(source, "encode_q4_expert_route_parallel_p1j").contains("Some(sidecar)"));
        assert!(
            treatment
                .find("validate_route_parallel_scratch_identity(")
                .unwrap()
                < treatment.find("encoder.begin_compute_pass(").unwrap()
        );
        for forbidden in [
            ".submit(",
            ".poll(",
            "map_async",
            "copy_buffer_to_buffer",
            "create_command_encoder",
            "rayon",
            "spawn",
        ] {
            assert!(!treatment.contains(forbidden), "{forbidden}");
        }
        assert!(!treatment.contains("for route_slot in"));
        assert!(treatment
            .contains("pass.dispatch_workgroups(gate_up_workgroups, geometry.route_width, 1)"));
        assert!(treatment
            .contains("pass.dispatch_workgroups(down_workgroups, geometry.route_width, 1)"));
        let serial = function(
            include_str!("../gpu_native.rs"),
            "encode_q4_expert_arena_combine",
        );
        assert!(serial.contains("for route_slot in 0..arena.geometry.top_k"));
        let combine = function(GPU_NATIVE_EXPERT_CONTROL_SHADER, "expert_combine_main");
        assert!(combine.contains("for (var route = 0u; route < pc.top_k; route += 1u)"));
        assert!(combine.contains("value += SELECTED_WEIGHTS[route]"));
        for method in [&serial, &treatment] {
            let mut prev = 0;
            for pass in [
                "gpu_native_expert_route_resolve_pass",
                "gpu_native_q4_expert_gate_up_pass",
                "gpu_native_q4_expert_down_pass",
                "gpu_native_expert_validate_pass",
                "gpu_native_expert_combine_pass",
                "gpu_native_expert_contain_pass",
                "self.encode_residual_add_pass",
            ] {
                let next = method.find(pass).unwrap();
                assert!(next > prev);
                prev = next;
            }
        }
    }

    // This test uses ONLY a CPU Vulkan adapter (lavapipe), never a hardware
    // GPU. It is explicit because portable installations need not ship Mesa.
    #[test]
    #[ignore = "requires a CPU Vulkan software adapter; never runs hardware"]
    fn q4_route_parallel_software_vulkan_bit_identity_and_failure_containment() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .into_iter()
            .find(|a| a.get_info().device_type == wgpu::DeviceType::Cpu)
            .expect("CPU Vulkan adapter required; hardware execution prohibited");
        assert_eq!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
        eprintln!("PR2-C software adapter: {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("PR2-C software only"),
                required_features: wgpu::Features::PUSH_CONSTANTS,
                required_limits: adapter.limits(),
            },
            None,
        ))
        .unwrap();
        for k in 1..=8 {
            let c = software_case(&device, &queue, false, k, 0, false, false, false);
            let t = software_case(&device, &queue, true, k, 0, false, false, false);
            assert_eq!(c, t, "same-device f32 bits must match at top_k={k}");
            assert_eq!(c.0, 0);
            assert!(c.2.iter().any(|&v| v != 0));
        }
        for (initial_status, stale_epoch, bad_numerical, private_output) in [
            (4, false, false, false),
            (0, true, false, false),
            (0, false, true, false),
            (8, false, false, true),
        ] {
            for parallel in [false, true] {
                let result = software_case(
                    &device,
                    &queue,
                    parallel,
                    8,
                    initial_status,
                    stale_epoch,
                    bad_numerical,
                    private_output,
                );
                assert_ne!(result.0, 0);
                assert!(
                    result.2.iter().all(|&bits| bits == 0),
                    "all mixture bytes must be contained"
                );
                if stale_epoch {
                    assert_eq!(result.0, 4);
                }
                if bad_numerical {
                    assert_eq!(result.0, 8);
                }
                if private_output {
                    assert!(result.1.iter().any(|&bits| bits != 0));
                }
            }
        }
    }

    fn software_case(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        parallel: bool,
        k: usize,
        initial_status: u32,
        stale_epoch: bool,
        numerical: bool,
        private_output: bool,
    ) -> (u32, Vec<u32>, Vec<u32>) {
        use wgpu::util::DeviceExt;
        let g = GpuNativeQ4ExpertGeometry::try_new(96, 160, 128, k).unwrap();
        let buffer = |label, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        let mut bank_bytes: [Vec<u8>; 4] =
            std::array::from_fn(|_| vec![0; g.slot_stride_bytes * k.div_ceil(4)]);
        let mut routes = Vec::<u32>::new();
        for route in 0..k {
            let bank = route % 4;
            let slot = route / 4;
            let start = slot * g.slot_stride_bytes;
            bank_bytes[bank][start..start + 4].copy_from_slice(&1u32.to_le_bytes());
            let mut payload = super::super::tests::q4_uniform_expert(
                g,
                0.015625 * (route + 1) as f32,
                -0.03125 * (route + 1) as f32,
                0.0078125 * (route + 1) as f32,
            );
            if numerical && route == k - 1 {
                payload[0..2].copy_from_slice(&half::f16::NAN.to_bits().to_le_bytes());
            }
            bank_bytes[bank][start + 4..start + 4 + payload.len()].copy_from_slice(&payload);
            routes.push(((bank as u32) << 30) | slot as u32);
            routes.push(if stale_epoch && route == k - 1 { 2 } else { 1 });
        }
        let banks: Vec<_> = bank_bytes.iter().map(|b| buffer("banks", b)).collect();
        let hidden: Vec<f32> = (0..g.d_model)
            .map(|i| ((i % 11) as f32 - 4.0) * 0.03125)
            .collect();
        let input = buffer("input", bytemuck::cast_slice(&hidden));
        let resolved = buffer("resolved", bytemuck::cast_slice(&routes));
        let activation = buffer(
            "activation",
            &vec![0; g.d_ff * if parallel { k } else { 1 } * 4],
        );
        let outputs = buffer(
            "outputs",
            bytemuck::cast_slice(&vec![
                if private_output { 3.0f32 } else { 0.0 };
                k * g.d_model
            ]),
        );
        let combined = buffer("combined", bytemuck::cast_slice(&vec![7.0f32; g.d_model]));
        let status = buffer("status", &initial_status.to_le_bytes());
        let weights: Vec<f32> = (0..k)
            .map(|i| (i + 1) as f32 / (k * (k + 1) / 2) as f32)
            .collect();
        let weights = buffer("weights", bytemuck::cast_slice(&weights));
        let binding_layout = |read_only: &[bool]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &read_only
                    .iter()
                    .enumerate()
                    .map(|(i, &read_only)| wgpu::BindGroupLayoutEntry {
                        binding: i as u32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let expert_layout = binding_layout(&[true, true, true, true, true, true, false, false]);
        let combine_layout = binding_layout(&[true, true, true, false, false]);
        let empty_layout = binding_layout(&[]);
        let empty_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &empty_layout,
            entries: &[],
        });
        let group = |layout: &wgpu::BindGroupLayout, buffers: &[&wgpu::Buffer]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &buffers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let gate_group = group(
            &expert_layout,
            &[
                &banks[0],
                &banks[1],
                &banks[2],
                &banks[3],
                &input,
                &resolved,
                &activation,
                &status,
            ],
        );
        let down_group = group(
            &expert_layout,
            &[
                &banks[0],
                &banks[1],
                &banks[2],
                &banks[3],
                &activation,
                &resolved,
                &outputs,
                &status,
            ],
        );
        let combine_group = group(
            &combine_layout,
            &[&weights, &resolved, &outputs, &combined, &status],
        );
        let expert_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                if parallel {
                    ROUTE_PARALLEL_SHADER
                } else {
                    GPU_NATIVE_Q4_EXPERT_SHADER
                }
                .into(),
            ),
        });
        let control_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_EXPERT_CONTROL_SHADER.into()),
        });
        let pipeline =
            |module: &wgpu::ShaderModule, entry_point, layouts: &[&wgpu::BindGroupLayout]| {
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: layouts,
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::COMPUTE,
                        range: 0..32,
                    }],
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&layout),
                    module,
                    entry_point,
                    compilation_options: Default::default(),
                })
            };
        let gate = pipeline(
            &expert_module,
            if parallel {
                "q4_expert_gate_up_route_parallel_main"
            } else {
                "q4_expert_gate_up_main"
            },
            &[&expert_layout],
        );
        let down = pipeline(
            &expert_module,
            if parallel {
                "q4_expert_down_route_parallel_main"
            } else {
                "q4_expert_down_main"
            },
            &[&expert_layout],
        );
        let validate = pipeline(
            &control_module,
            "expert_validate_main",
            &[&empty_layout, &combine_layout],
        );
        let combine = pipeline(
            &control_module,
            "expert_combine_main",
            &[&empty_layout, &combine_layout],
        );
        let contain = pipeline(
            &control_module,
            "expert_contain_main",
            &[&empty_layout, &combine_layout],
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        if !private_output {
            for route in 0..if parallel { 1 } else { k } {
                let pc = GpuNativeQ4ExpertPushConstants {
                    d_model: g.d_model as u32,
                    d_ff: g.d_ff as u32,
                    blocks_per_projection: g.blocks_per_projection as u32,
                    slot_stride_bytes: g.slot_stride_bytes as u32,
                    route_slot: route as u32,
                    top_k: k as u32,
                    swiglu_limit: 3.0,
                    _reserved: 0,
                };
                for (pipeline, group, rows) in [
                    (&gate, &gate_group, g.d_ff),
                    (&down, &down_group, g.d_model),
                ] {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.set_push_constants(0, bytemuck::bytes_of(&pc));
                    pass.dispatch_workgroups(
                        rows.div_ceil(64) as u32,
                        if parallel { k as u32 } else { 1 },
                        1,
                    );
                }
            }
        }
        let pc: [u32; 8] = [g.d_model as u32, k as u32, 0, 0, 0, 0, 0, 0];
        for (pipeline, x) in [
            (&validate, 1),
            (&combine, g.d_model.div_ceil(64) as u32),
            (&contain, g.d_model.div_ceil(64) as u32),
        ] {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &empty_group, &[]);
            pass.set_bind_group(1, &combine_group, &[]);
            pass.set_push_constants(0, bytemuck::cast_slice(&pc));
            pass.dispatch_workgroups(x, 1, 1);
        }
        let output_bytes = (k * g.d_model * 4) as u64;
        let combined_bytes = (g.d_model * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4 + output_bytes + combined_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&status, 0, &staging, 0, 4);
        encoder.copy_buffer_to_buffer(&outputs, 0, &staging, 4, output_bytes);
        encoder.copy_buffer_to_buffer(&combined, 0, &staging, 4 + output_bytes, combined_bytes);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let mapped = staging.slice(..).get_mapped_range();
        let words: Vec<u32> = mapped
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        (
            words[0],
            words[1..1 + k * g.d_model].to_vec(),
            words[1 + k * g.d_model..].to_vec(),
        )
    }
}
