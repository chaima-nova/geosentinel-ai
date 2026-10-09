import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// No <StrictMode>: react-leaflet's MapContainer double-mounts under it in dev,
// which trips Leaflet's "container is already initialized" guard.
ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
