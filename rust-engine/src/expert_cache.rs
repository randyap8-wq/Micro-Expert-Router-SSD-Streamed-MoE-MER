use std::sync::Arc;
use tokio::sync::Mutex;
use lru::LruCache;
use std::num::NonZeroUsize;

pub struct ExpertData {
    pub id: u32,
    pub buffer: Vec<u8>, // Scaled for O_DIRECT alignment
}

pub struct ExpertCache {
    cache: Arc<Mutex<LruCache<u32, Arc<ExpertData>>>>,
    capacity: usize,
}

impl ExpertCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(capacity).unwrap()))),
            capacity,
        }
    }

    pub async fn get(&self, id: u32) -> Option<Arc<ExpertData>> {
        let mut cache = self.cache.lock().await;
        cache.get(&id).cloned()
    }

    pub async fn insert(&self, id: u32, data: Arc<ExpertData>) {
        let mut cache = self.cache.lock().await;
        cache.put(id, data);
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
