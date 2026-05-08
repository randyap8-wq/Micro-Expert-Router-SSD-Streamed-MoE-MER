mod expert_cache;
mod io_provider;
mod router;

use crate::expert_cache::ExpertCache;
use crate::io_provider::NVMeStorage;
use crate::router::PredictiveRouter;
use std::sync::Arc;
use tokio::time::{Instant, Duration};

#[tokio::main]
async fn main() {
    println!("--- Industrial Micro-Expert Router Initializing ---");
    
    // Config: 64 experts of 128MB each, 8GB RAM cache (64 slots)
    let expert_size = 128 * 1024 * 1024;
    let cache_capacity = 32; 
    
    let cache = Arc::new(ExpertCache::new(cache_capacity));
    let storage = Arc::new(NVMeStorage::new("./data", expert_size));
    let router = PredictiveRouter::new(64);

    println!("SSD Hot-Swap Engine Running...");
    println!("Cache Capacity: {} Experts", cache_capacity);

    // Simulation Loop
    let mut last_id = 0;
    for token in 0..100 {
        let start = Instant::now();
        let target_experts = router.route_request();
        
        println!("\n[Token {}] Routing to Experts {:?}", token, target_experts);

        for &id in &target_experts {
            if let Some(_data) = cache.get(id).await {
                println!("  [ID {}] CACHE HIT", id);
            } else {
                println!("  [ID {}] CACHE MISS - Fetching from NVMe via io_uring...", id);
                let io_start = Instant::now();
                match storage.read_expert(id).await {
                    Ok(data) => {
                        cache.insert(id, data).await;
                        println!("    IO Latency: {:?}", io_start.elapsed());
                    }
                    Err(e) => println!("    IO ERROR: {:?}", e),
                }
            }
            last_id = id;
        }

        // Predictive Loading (Background)
        let predictions = router.predict_next(last_id);
        for pred_id in predictions {
            if cache.get(pred_id).await.is_none() {
                let storage_c = Arc::clone(&storage);
                let cache_c = Arc::clone(&cache);
                tokio::spawn(async move {
                    if let Ok(data) = storage_c.read_expert(pred_id).await {
                        cache_c.insert(pred_id, data).await;
                    }
                });
            }
        }

        let latency = start.elapsed();
        let throughput = 1.0 / latency.as_secs_f64();
        println!("Cycle Latency: {:?} | Approx Target Throughput: {:.2} tokens/sec", latency, throughput);
        
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}
