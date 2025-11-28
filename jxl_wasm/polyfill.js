/**
 * JXL Polyfill - Automatic JXL support for browsers
 *
 * Usage:
 *   <script type="module" src="polyfill.js"></script>
 *
 * Features:
 * - Automatically converts <img src="*.jxl"> tags
 * - Watches for dynamically added images
 * - Patches Image() constructor (optional)
 * - Shows loading state during conversion
 */

import init, { decode_jxl_to_png, init_panic_hook } from './pkg/jxl_wasm.js';

class JXLPolyfill {
    constructor(options = {}) {
        this.options = {
            patchImageConstructor: options.patchImageConstructor ?? true,
            showLoadingState: options.showLoadingState ?? true,
            cacheDecoded: options.cacheDecoded ?? true,
            verbose: options.verbose ?? false,
            ...options
        };

        this.initialized = false;
        this.initPromise = null;
        this.cache = new Map(); // URL -> Blob URL
        this.processing = new Set(); // URLs currently being processed
        this.observer = null;
    }

    async init() {
        if (this.initialized) return;
        if (this.initPromise) return this.initPromise;

        this.initPromise = (async () => {
            await init();
            init_panic_hook();
            this.initialized = true;
            this.log('JXL Polyfill initialized');
        })();

        return this.initPromise;
    }

    log(...args) {
        if (this.options.verbose) {
            console.log('[JXL Polyfill]', ...args);
        }
    }

    isJXL(url) {
        if (!url) return false;
        const urlStr = url.toString().toLowerCase();
        return urlStr.endsWith('.jxl') || urlStr.includes('.jxl?');
    }

    async convertJXL(url) {
        // Check cache
        if (this.options.cacheDecoded && this.cache.has(url)) {
            this.log('Using cached:', url);
            return this.cache.get(url);
        }

        // Prevent duplicate processing
        if (this.processing.has(url)) {
            this.log('Already processing:', url);
            // Wait for existing conversion
            return new Promise((resolve) => {
                const check = setInterval(() => {
                    if (!this.processing.has(url)) {
                        clearInterval(check);
                        resolve(this.cache.get(url));
                    }
                }, 100);
            });
        }

        this.processing.add(url);

        try {
            this.log('Converting:', url);

            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const jxlData = new Uint8Array(await response.arrayBuffer());
            const pngData = decode_jxl_to_png(jxlData);
            const blob = new Blob([pngData], { type: 'image/png' });
            const blobUrl = URL.createObjectURL(blob);

            if (this.options.cacheDecoded) {
                this.cache.set(url, blobUrl);
            }

            this.log('Converted:', url, '->', blobUrl);
            return blobUrl;

        } catch (error) {
            console.error('[JXL Polyfill] Conversion failed:', url, error);
            throw error;
        } finally {
            this.processing.delete(url);
        }
    }

    async processImage(img) {
        const src = img.getAttribute('src') || img.src;
        if (!src || !this.isJXL(src)) return;

        // Skip if already processed
        if (img.dataset.jxlProcessed) return;
        img.dataset.jxlProcessed = 'processing';

        const originalSrc = src;

        try {
            // Show loading state
            if (this.options.showLoadingState) {
                img.style.opacity = '0.5';
                img.style.filter = 'blur(2px)';
            }

            await this.init();
            const pngUrl = await this.convertJXL(src);

            // Update image
            img.src = pngUrl;
            img.dataset.jxlProcessed = 'true';
            img.dataset.jxlOriginal = originalSrc;

            // Remove loading state
            if (this.options.showLoadingState) {
                img.style.opacity = '';
                img.style.filter = '';
            }

        } catch (error) {
            img.dataset.jxlProcessed = 'error';
            img.alt = `Failed to load JXL: ${error.message}`;
            console.error('[JXL Polyfill] Failed to process image:', img, error);
        }
    }

    scanAndConvert() {
        const images = document.querySelectorAll('img');
        this.log('Scanning', images.length, 'images');

        images.forEach(img => {
            if (this.isJXL(img.src || img.getAttribute('src'))) {
                this.processImage(img);
            }
        });
    }

    observeDOM() {
        if (this.observer) return;

        this.observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                // Check added nodes
                for (const node of mutation.addedNodes) {
                    if (node.nodeType === 1) { // Element
                        if (node.tagName === 'IMG') {
                            this.processImage(node);
                        }
                        // Check children
                        const imgs = node.querySelectorAll?.('img') || [];
                        imgs.forEach(img => this.processImage(img));
                    }
                }

                // Check attribute changes on img tags
                if (mutation.type === 'attributes' &&
                    mutation.target.tagName === 'IMG' &&
                    mutation.attributeName === 'src') {
                    const img = mutation.target;
                    // Reset processed flag when src changes
                    delete img.dataset.jxlProcessed;
                    this.processImage(img);
                }
            }
        });

        this.observer.observe(document.documentElement, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['src']
        });

        this.log('DOM observer started');
    }

    patchImage() {
        if (!this.options.patchImageConstructor) return;

        const OriginalImage = window.Image;
        const polyfill = this;

        window.Image = class extends OriginalImage {
            constructor(width, height) {
                super(width, height);

                const originalSetSrc = Object.getOwnPropertyDescriptor(
                    HTMLImageElement.prototype, 'src'
                ).set;

                let currentSrc = '';

                Object.defineProperty(this, 'src', {
                    get() {
                        return currentSrc;
                    },
                    set(value) {
                        currentSrc = value;

                        if (polyfill.isJXL(value)) {
                            polyfill.log('Image() constructor intercepted:', value);

                            // Set to data URL initially to prevent 404
                            originalSetSrc.call(this, 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7');

                            polyfill.init().then(() => {
                                polyfill.convertJXL(value).then(pngUrl => {
                                    originalSetSrc.call(this, pngUrl);
                                }).catch(error => {
                                    console.error('[JXL Polyfill] Image() conversion failed:', error);
                                });
                            });
                        } else {
                            originalSetSrc.call(this, value);
                        }
                    }
                });
            }
        };

        // Preserve constructor name
        Object.defineProperty(window.Image, 'name', { value: 'Image' });

        this.log('Image() constructor patched');
    }

    start() {
        // Wait for DOM ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.start());
            return;
        }

        this.log('Starting polyfill');

        // Patch Image constructor first
        this.patchImage();

        // Scan existing images
        this.scanAndConvert();

        // Watch for new images
        this.observeDOM();
    }

    stop() {
        if (this.observer) {
            this.observer.disconnect();
            this.observer = null;
            this.log('DOM observer stopped');
        }
    }

    clearCache() {
        // Revoke all blob URLs
        for (const blobUrl of this.cache.values()) {
            URL.revokeObjectURL(blobUrl);
        }
        this.cache.clear();
        this.log('Cache cleared');
    }
}

// Auto-start by default
const polyfill = new JXLPolyfill({
    verbose: new URLSearchParams(window.location.search).has('jxl-debug')
});

polyfill.start();

// Export for manual control
export default polyfill;
export { JXLPolyfill };
