/**
 * JXL Polyfill - Automatic JXL support for browsers
 *
 * Usage:
 *   <script type="module" src="polyfill.js"></script>
 *
 * Features:
 * - Automatically converts <img src="*.jxl"> tags
 * - Supports CSS background-image with JXL URLs
 * - Supports <source srcset="*.jxl"> in <picture> elements
 * - Supports SVG <image> and <feImage> elements with JXL URLs
 * - Watches for dynamically added elements
 * - Patches Image() constructor (optional)
 * - Shows loading state during conversion
 * - Caches decoded images for performance
 */

import init, { decode_jxl_to_png, init_panic_hook } from './pkg/jxl_wasm.js';

class JXLPolyfill {
    constructor(options = {}) {
        this.options = {
            patchImageConstructor: options.patchImageConstructor ?? true,
            showLoadingState: options.showLoadingState ?? true,
            cacheDecoded: options.cacheDecoded ?? true,
            verbose: options.verbose ?? false,
            // New options for extended support
            handleCSSBackgrounds: options.handleCSSBackgrounds ?? true,
            handleSourceElements: options.handleSourceElements ?? true,
            handleSVGElements: options.handleSVGElements ?? true,
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
        return urlStr.endsWith('.jxl') || urlStr.includes('.jxl?') || urlStr.includes('.jxl#');
    }

    /**
     * Extract JXL URL from CSS background-image value
     * @param {string} bgValue - CSS background-image value like 'url("image.jxl")'
     * @returns {string|null} - The URL or null if not a JXL
     */
    extractJXLFromBackground(bgValue) {
        if (!bgValue || bgValue === 'none') return null;

        // Match url(...) patterns
        const match = bgValue.match(/url\(["']?([^"')]+\.jxl[^"')]*?)["']?\)/i);
        if (match && match[1]) {
            return match[1];
        }
        return null;
    }

    async convertJXL(url) {
        // Resolve relative URLs to absolute
        const absoluteUrl = new URL(url, window.location.href).href;

        // Check cache
        if (this.options.cacheDecoded && this.cache.has(absoluteUrl)) {
            this.log('Using cached:', absoluteUrl);
            return this.cache.get(absoluteUrl);
        }

        // Prevent duplicate processing
        if (this.processing.has(absoluteUrl)) {
            this.log('Already processing:', absoluteUrl);
            // Wait for existing conversion
            return new Promise((resolve) => {
                const check = setInterval(() => {
                    if (!this.processing.has(absoluteUrl)) {
                        clearInterval(check);
                        resolve(this.cache.get(absoluteUrl));
                    }
                }, 100);
            });
        }

        this.processing.add(absoluteUrl);

        try {
            this.log('Converting:', absoluteUrl);

            const response = await fetch(absoluteUrl);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const jxlData = new Uint8Array(await response.arrayBuffer());
            const pngData = decode_jxl_to_png(jxlData);
            const blob = new Blob([pngData], { type: 'image/png' });
            const blobUrl = URL.createObjectURL(blob);

            if (this.options.cacheDecoded) {
                this.cache.set(absoluteUrl, blobUrl);
            }

            this.log('Converted:', absoluteUrl, '->', blobUrl);
            return blobUrl;

        } catch (error) {
            console.error('[JXL Polyfill] Conversion failed:', absoluteUrl, error);
            throw error;
        } finally {
            this.processing.delete(absoluteUrl);
        }
    }

    // ==================== IMG Element Support ====================

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

    // ==================== CSS Background Support ====================

    async processBackgroundImage(element) {
        if (!this.options.handleCSSBackgrounds) return;

        const computedStyle = getComputedStyle(element);
        const bgImage = computedStyle.backgroundImage;
        const jxlUrl = this.extractJXLFromBackground(bgImage);

        if (!jxlUrl) return;

        // Skip if already processed
        if (element.dataset.jxlBgProcessed) return;
        element.dataset.jxlBgProcessed = 'processing';

        try {
            await this.init();
            const pngUrl = await this.convertJXL(jxlUrl);

            // Update background-image
            element.style.backgroundImage = `url("${pngUrl}")`;
            element.dataset.jxlBgProcessed = 'true';
            element.dataset.jxlBgOriginal = jxlUrl;

            this.log('Updated background-image:', element, jxlUrl, '->', pngUrl);

        } catch (error) {
            element.dataset.jxlBgProcessed = 'error';
            console.error('[JXL Polyfill] Failed to process background:', element, error);
        }
    }

    // ==================== <source> Element Support ====================

    async processSourceElement(source) {
        if (!this.options.handleSourceElements) return;

        const srcset = source.getAttribute('srcset');
        if (!srcset || !this.isJXL(srcset)) return;

        // Skip if already processed
        if (source.dataset.jxlProcessed) return;
        source.dataset.jxlProcessed = 'processing';

        try {
            await this.init();

            // Parse srcset (can have multiple entries like "image.jxl 1x, image2.jxl 2x")
            const entries = srcset.split(',').map(s => s.trim());
            const newEntries = [];

            for (const entry of entries) {
                const parts = entry.split(/\s+/);
                const url = parts[0];
                const descriptor = parts.slice(1).join(' ');

                if (this.isJXL(url)) {
                    const pngUrl = await this.convertJXL(url);
                    newEntries.push(descriptor ? `${pngUrl} ${descriptor}` : pngUrl);
                } else {
                    newEntries.push(entry);
                }
            }

            // Update srcset
            source.srcset = newEntries.join(', ');
            source.type = 'image/png';
            source.dataset.jxlProcessed = 'true';
            source.dataset.jxlOriginal = srcset;

            this.log('Updated <source> srcset:', source);

        } catch (error) {
            source.dataset.jxlProcessed = 'error';
            console.error('[JXL Polyfill] Failed to process <source>:', source, error);
        }
    }

    // ==================== SVG Element Support ====================

    async processSVGImage(svgElement) {
        if (!this.options.handleSVGElements) return;

        // SVG uses href or xlink:href
        const href = svgElement.getAttribute('href') ||
                     svgElement.getAttributeNS('http://www.w3.org/1999/xlink', 'href');

        if (!href || !this.isJXL(href)) return;

        // Skip if already processed
        if (svgElement.dataset.jxlProcessed) return;
        svgElement.dataset.jxlProcessed = 'processing';

        try {
            await this.init();
            const pngUrl = await this.convertJXL(href);

            // Update href (prefer standard href over xlink:href)
            if (svgElement.hasAttribute('href')) {
                svgElement.setAttribute('href', pngUrl);
            } else {
                svgElement.setAttributeNS('http://www.w3.org/1999/xlink', 'href', pngUrl);
            }

            svgElement.dataset.jxlProcessed = 'true';
            svgElement.dataset.jxlOriginal = href;

            this.log('Updated SVG element:', svgElement.tagName, href, '->', pngUrl);

        } catch (error) {
            svgElement.dataset.jxlProcessed = 'error';
            console.error('[JXL Polyfill] Failed to process SVG element:', svgElement, error);
        }
    }

    // ==================== Scanning ====================

    scanAndConvert() {
        this.log('Scanning for JXL images...');

        // Scan <img> elements
        const images = document.querySelectorAll('img');
        this.log('Found', images.length, 'img elements');
        images.forEach(img => {
            if (this.isJXL(img.src || img.getAttribute('src'))) {
                this.processImage(img);
            }
        });

        // Scan <source> elements in <picture>
        if (this.options.handleSourceElements) {
            const sources = document.querySelectorAll('picture source');
            this.log('Found', sources.length, 'source elements');
            sources.forEach(source => {
                if (this.isJXL(source.srcset || source.getAttribute('srcset'))) {
                    this.processSourceElement(source);
                }
            });
        }

        // Scan SVG <image> and <feImage> elements
        if (this.options.handleSVGElements) {
            const svgImages = document.querySelectorAll('image, feImage');
            this.log('Found', svgImages.length, 'SVG image elements');
            svgImages.forEach(el => {
                const href = el.getAttribute('href') ||
                             el.getAttributeNS('http://www.w3.org/1999/xlink', 'href');
                if (this.isJXL(href)) {
                    this.processSVGImage(el);
                }
            });
        }

        // Scan elements with CSS background-image
        if (this.options.handleCSSBackgrounds) {
            // Get all elements and check their computed styles
            const allElements = document.querySelectorAll('*');
            let bgCount = 0;
            allElements.forEach(el => {
                const bgImage = getComputedStyle(el).backgroundImage;
                if (this.extractJXLFromBackground(bgImage)) {
                    bgCount++;
                    this.processBackgroundImage(el);
                }
            });
            this.log('Found', bgCount, 'elements with JXL backgrounds');
        }
    }

    // ==================== DOM Observer ====================

    observeDOM() {
        if (this.observer) return;

        this.observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                // Check added nodes
                for (const node of mutation.addedNodes) {
                    if (node.nodeType !== 1) continue; // Element nodes only

                    // Check the node itself
                    this.checkElement(node);

                    // Check children
                    if (node.querySelectorAll) {
                        node.querySelectorAll('img, source, image, feImage, [style*="background"]')
                            .forEach(el => this.checkElement(el));
                    }
                }

                // Check attribute changes
                if (mutation.type === 'attributes' && mutation.target.nodeType === 1) {
                    const target = mutation.target;
                    const attr = mutation.attributeName;

                    if (attr === 'src' && target.tagName === 'IMG') {
                        delete target.dataset.jxlProcessed;
                        this.processImage(target);
                    } else if (attr === 'srcset' && target.tagName === 'SOURCE') {
                        delete target.dataset.jxlProcessed;
                        this.processSourceElement(target);
                    } else if ((attr === 'href' || attr === 'xlink:href') &&
                               (target.tagName === 'image' || target.tagName === 'feImage')) {
                        delete target.dataset.jxlProcessed;
                        this.processSVGImage(target);
                    } else if (attr === 'style') {
                        delete target.dataset.jxlBgProcessed;
                        this.processBackgroundImage(target);
                    }
                }
            }
        });

        this.observer.observe(document.documentElement, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['src', 'srcset', 'href', 'xlink:href', 'style']
        });

        this.log('DOM observer started');
    }

    checkElement(el) {
        const tagName = el.tagName?.toUpperCase();

        if (tagName === 'IMG') {
            this.processImage(el);
        } else if (tagName === 'SOURCE') {
            this.processSourceElement(el);
        } else if (tagName === 'IMAGE' || tagName === 'FEIMAGE') {
            this.processSVGImage(el);
        }

        // Check for background-image
        if (this.options.handleCSSBackgrounds) {
            const bgImage = getComputedStyle(el).backgroundImage;
            if (this.extractJXLFromBackground(bgImage)) {
                this.processBackgroundImage(el);
            }
        }
    }

    // ==================== Image Constructor Patch ====================

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

    // ==================== Lifecycle ====================

    start() {
        // Wait for DOM ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.start());
            return;
        }

        this.log('Starting polyfill');

        // Patch Image constructor first
        this.patchImage();

        // Scan existing elements
        this.scanAndConvert();

        // Watch for new elements
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

    /**
     * Get statistics about the polyfill state
     */
    getStats() {
        return {
            initialized: this.initialized,
            cacheSize: this.cache.size,
            processingCount: this.processing.size,
            cachedUrls: [...this.cache.keys()]
        };
    }
}

// Check for native JXL support before starting polyfill
async function checkNativeJXLSupport() {
    // Create a 1x1 JXL image (smallest valid JXL)
    const jxlData = 'data:image/jxl;base64,/woIELASCAgQAFzgBzgBPAk=';

    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(true);
        img.onerror = () => resolve(false);
        img.src = jxlData;

        // Timeout after 100ms
        setTimeout(() => resolve(false), 100);
    });
}

// Auto-start only if native support is not available
checkNativeJXLSupport().then(hasNativeSupport => {
    if (hasNativeSupport) {
        console.log('[JXL Polyfill] Native JXL support detected, polyfill not needed');
        return;
    }

    console.log('[JXL Polyfill] No native support, loading polyfill');
    const polyfill = new JXLPolyfill({
        verbose: new URLSearchParams(window.location.search).has('jxl-debug')
    });

    polyfill.start();

    // Export for manual control
    window.jxlPolyfill = polyfill;
});

// Export for manual control
export { JXLPolyfill };
