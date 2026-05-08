use rand::Rng;
use std::collections::HashMap;

pub struct PredictiveRouter {
    // Markov-style transition matrix [CurrentExpertID][NextExpertID] -> Probability
    weights: HashMap<u32, Vec<(u32, f64)>>,
}

impl PredictiveRouter {
    pub fn new(num_experts: u32) -> Self {
        let mut weights = HashMap::new();
        for i in 0..num_experts {
            let mut transitions = Vec::new();
            // Generate some random temporal locality for simulation
            for j in 0..num_experts {
                if (i as i32 - j as i32).abs() < 5 {
                    transitions.push((j, 0.2));
                } else {
                    transitions.push((j, 0.01));
                }
            }
            weights.insert(i, transitions);
        }
        Self { weights }
    }

    pub fn predict_next(&self, last_expert: u32) -> Vec<u32> {
        if let Some(transitions) = self.weights.get(&last_expert) {
            let mut sorted = transitions.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            sorted.iter().take(2).map(|(id, _)| *id).collect()
        } else {
            vec![]
        }
    }

    pub fn route_request(&self) -> Vec<u32> {
        let mut rng = rand::thread_rng();
        // Mimic Top-K routing (K=2)
        vec![rng.gen_range(0..64), rng.gen_range(0..64)]
    }
}
