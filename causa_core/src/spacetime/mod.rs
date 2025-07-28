// Declare the event and manifold modules
pub mod event;
pub mod manifold;

// Re-export the structs to make them easily accessible
pub use event::Event;
pub use manifold::Manifold;
