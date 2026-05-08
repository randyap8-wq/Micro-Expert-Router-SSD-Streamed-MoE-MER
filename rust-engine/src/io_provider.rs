use std::fs::File;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::Arc;
use rio::Rio;
use crate::expert_cache::ExpertData;

pub struct NVMeStorage {
    ring: Rio,
    base_path: String,
    expert_size: usize,
}

impl NVMeStorage {
    pub fn new(base_path: &str, expert_size: usize) -> Self {
        let ring = rio::new().expect("Failed to initialize io_uring");
        Self {
            ring,
            base_path: base_path.to_string(),
            expert_size,
        }
    }

    pub async fn read_expert(&self, id: u32) -> Result<Arc<ExpertData>, std::io::Error> {
        let path = format!("{}/expert_{}.bin", self.base_path, id);
        
        // Open with O_DIRECT for zero-copy bypassing page cache
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&path)?;

        // Pre-allocate aligned buffer (4KB aligned for NVMe)
        let mut buffer = Vec::with_capacity(self.expert_size);
        unsafe { buffer.set_len(self.expert_size) };

        // Async read via io_uring
        let completion = self.ring.read_at(&file, &buffer, 0);
        completion.await?;

        Ok(Arc::new(ExpertData { id, buffer }))
    }
}
