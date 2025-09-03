
import type { MetadataRoute } from "next";
 
export default function manifest(): MetadataRoute.Manifest {
  return {
    "name": "App",
    "short_name": "App",
    "start_url": "/",
    "background_color": "#2a3a46",
    "theme_color": "#2a3a46",
    "display": "standalone",
    "icons": [       
        {
            "src": "images/pwa-logo.webp",
            "sizes": "512x512",
            "type": "image/webp"
        },
        {
            "src": "images/pwa-logo-small.webp",
            "sizes": "192x192",
            "type": "image/webp"
        },
    ],
  };
};