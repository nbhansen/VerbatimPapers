/**
 * Enhanced Read-Along Player with audiobook-quality features
 */
class EnhancedReadAlongPlayer {
    constructor(timingApiUrl, audioElement) {
        this.timingApiUrl = timingApiUrl;
        this.audio = audioElement;
        this.segments = [];
        this.currentSegmentIndex = -1;
        this.autoScroll = true;
        this.isUserSeeking = false;

        // Enhanced features
        this.readingSpeed = 1.0;
        this.highlightMode = 'smooth'; // 'smooth' or 'instant'
        this.lookaheadSegments = 2; // Pre-highlight upcoming segments
        this.scrollOffset = 0.3; // Scroll when text is 30% from bottom

        // Performance optimization
        this.updateInterval = null;
        this.lastUpdateTime = 0;
        this.updateFrequency = 50; // ms

        // Visual feedback
        this.confidenceThreshold = 0.7;
        this.wordHighlighting = false;

        this.initializeElements();
        this.loadTimingData();
        this.setupEventListeners();
        this.startUpdateLoop();
    }

    initializeElements() {
        this.textContainer = document.getElementById('textContainer');
        this.currentTimeDisplay = document.getElementById('currentTime');
        this.totalTimeDisplay = document.getElementById('totalTime');
        this.toggleAutoScrollBtn = document.getElementById('toggleAutoScroll');
        this.resumeTrackingBtn = document.getElementById('resumeTracking');

        // Add new controls
        this.createEnhancedControls();
    }

    createEnhancedControls() {
        const controlsContainer = document.querySelector('.sync-controls');

        // Speed control
        const speedControl = document.createElement('select');
        speedControl.id = 'speedControl';
        speedControl.className = 'control-btn';
        speedControl.innerHTML = `
            <option value="0.75">0.75x</option>
            <option value="1" selected>1x</option>
            <option value="1.25">1.25x</option>
            <option value="1.5">1.5x</option>
        `;
        speedControl.addEventListener('change', (e) => {
            this.setPlaybackSpeed(parseFloat(e.target.value));
        });

        // Highlight mode toggle
        const highlightToggle = document.createElement('button');
        highlightToggle.id = 'highlightMode';
        highlightToggle.className = 'control-btn';
        highlightToggle.textContent = 'SMOOTH HIGHLIGHT';
        highlightToggle.addEventListener('click', () => {
            this.toggleHighlightMode();
        });

        controlsContainer.appendChild(speedControl);
        controlsContainer.appendChild(highlightToggle);
    }

    async loadTimingData() {
        try {
            console.log('Loading timing data from:', this.timingApiUrl);
            const response = await fetch(this.timingApiUrl);
            console.log('Response status:', response.status);

            if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

            const timingData = await response.json();
            console.log('Timing data loaded:', timingData);

            this.segments = this.processTimingData(timingData.text_segments || []);
            this.renderText();

            console.log(`Enhanced player loaded ${this.segments.length} segments`);

            // Analyze timing quality
            this.analyzeTimingQuality();

        } catch (error) {
            console.error('Failed to load timing data:', error);
            this.showError(error.message);
        }
    }

    processTimingData(segments) {
        // Enhance segments with additional metadata
        return segments.map((segment, index) => ({
            ...segment,
            index,
            confidence: this.estimateConfidence(segment),
            wordBoundaries: this.estimateWordBoundaries(segment),
            nextSegmentGap: index < segments.length - 1
                ? segments[index + 1].start_time - (segment.start_time + segment.duration)
                : 0
        }));
    }

    estimateConfidence(segment) {
        // Estimate timing confidence based on segment characteristics
        let confidence = 0.8; // Base confidence

        // Adjust based on segment type
        if (segment.segment_type === 'technical') confidence -= 0.1;
        if (segment.segment_type === 'emphasis') confidence += 0.05;

        // Adjust based on duration reasonableness
        const wordsPerSecond = segment.text.split(/\s+/).length / segment.duration;
        if (wordsPerSecond < 1 || wordsPerSecond > 4) confidence -= 0.2;

        return Math.max(0.3, Math.min(1.0, confidence));
    }

    estimateWordBoundaries(segment) {
        // Estimate word timings within segment
        const words = segment.text.split(/\s+/);
        const wordDuration = segment.duration / words.length;

        return words.map((word, i) => ({
            word,
            start: segment.start_time + (i * wordDuration),
            end: segment.start_time + ((i + 1) * wordDuration)
        }));
    }

    renderText() {
        if (this.segments.length === 0) {
            this.textContainer.innerHTML = '<p style="color: #FBFE65;">No text segments available.</p>';
            return;
        }

        // Group segments by type for better visual organization
        const groupedHtml = this.segments.map((segment, index) => {
            const confidenceClass = segment.confidence > this.confidenceThreshold
                ? 'high-confidence'
                : 'low-confidence';

            const typeClass = `segment-type-${segment.segment_type}`;

            return `
                <span class="text-segment ${confidenceClass} ${typeClass}"
                      data-index="${index}"
                      data-start="${segment.start_time}"
                      data-confidence="${segment.confidence.toFixed(2)}">
                    ${this.escapeHtml(segment.text)}
                </span>
            `;
        }).join(' ');

        this.textContainer.innerHTML = `
            <div class="timing-quality-indicator">
                <span id="timingQuality"></span>
            </div>
            <div class="text-content">
                ${groupedHtml}
            </div>
        `;

        // Add click listeners
        this.textContainer.querySelectorAll('.text-segment').forEach(segment => {
            segment.addEventListener('click', (e) => {
                const startTime = parseFloat(e.target.dataset.start);
                this.seekToTime(startTime);
            });

            // Show confidence on hover
            segment.addEventListener('mouseenter', (e) => {
                const confidence = e.target.dataset.confidence;
                this.showTooltip(e.target, `Timing confidence: ${(confidence * 100).toFixed(0)}%`);
            });
        });
    }

    startUpdateLoop() {
        // High-frequency update loop for smooth highlighting
        this.updateInterval = setInterval(() => {
            const now = performance.now();
            if (now - this.lastUpdateTime >= this.updateFrequency) {
                this.updateHighlight();
                this.lastUpdateTime = now;
            }
        }, 16); // 60fps
    }

    updateHighlight() {
        if (this.isUserSeeking) return;

        const currentTime = this.audio.currentTime;
        const playbackRate = this.audio.playbackRate;

        // Find current and upcoming segments
        const activeSegments = this.findActiveSegments(currentTime, playbackRate);

        // Update highlighting with smooth transitions
        this.textContainer.querySelectorAll('.text-segment').forEach((element, index) => {
            const segment = this.segments[index];

            element.classList.remove('active', 'completed', 'upcoming', 'pre-active');

            if (activeSegments.current === index) {
                element.classList.add('active');

                // Smooth highlight progress
                if (this.highlightMode === 'smooth') {
                    const progress = (currentTime - segment.start_time) / segment.duration;
                    element.style.setProperty('--highlight-progress', progress);
                }

            } else if (index < activeSegments.current) {
                element.classList.add('completed');

            } else if (activeSegments.upcoming.includes(index)) {
                element.classList.add('upcoming');
                const distance = index - activeSegments.current;
                element.style.setProperty('--upcoming-distance', distance);
            }

            // Pre-active state for smoother transitions
            if (activeSegments.preActive === index) {
                element.classList.add('pre-active');
            }
        });

        // Handle scrolling
        if (activeSegments.current !== this.currentSegmentIndex && this.autoScroll) {
            this.scrollToSegment(activeSegments.current);
        }

        this.currentSegmentIndex = activeSegments.current;
        this.updateTimeDisplay();
    }

    findActiveSegments(currentTime, playbackRate) {
        let current = -1;
        let preActive = -1;
        const upcoming = [];

        // Account for playback rate in lookahead
        const lookaheadTime = this.lookaheadSegments * 0.5 / playbackRate;

        for (let i = 0; i < this.segments.length; i++) {
            const segment = this.segments[i];

            if (currentTime >= segment.start_time &&
                currentTime < segment.start_time + segment.duration) {
                current = i;

                // Check if we're near the end of current segment
                const timeToEnd = (segment.start_time + segment.duration) - currentTime;
                if (timeToEnd < 0.2 && i < this.segments.length - 1) {
                    preActive = i + 1;
                }

            } else if (segment.start_time > currentTime &&
                       segment.start_time < currentTime + lookaheadTime) {
                upcoming.push(i);
            }
        }

        return { current, upcoming, preActive };
    }

    scrollToSegment(index) {
        if (index < 0 || index >= this.segments.length) return;

        const element = this.textContainer.querySelector(`[data-index="${index}"]`);
        if (!element) return;

        const container = this.textContainer;
        const elementRect = element.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();

        // Calculate if scrolling is needed
        const relativeTop = elementRect.top - containerRect.top;
        const relativeBottom = elementRect.bottom - containerRect.top;
        const containerHeight = containerRect.height;

        // Scroll if element is in bottom 30% of container
        if (relativeBottom > containerHeight * (1 - this.scrollOffset)) {
            element.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }
    }

    setPlaybackSpeed(speed) {
        this.readingSpeed = speed;
        this.audio.playbackRate = speed;

        // Adjust timing data for new speed
        this.updateFrequency = Math.max(16, 50 / speed);

        console.log(`Playback speed set to ${speed}x`);
    }

    toggleHighlightMode() {
        this.highlightMode = this.highlightMode === 'smooth' ? 'instant' : 'smooth';
        const btn = document.getElementById('highlightMode');
        btn.textContent = this.highlightMode === 'smooth' ? 'SMOOTH HIGHLIGHT' : 'INSTANT HIGHLIGHT';

        // Update CSS class
        this.textContainer.classList.toggle('smooth-highlighting', this.highlightMode === 'smooth');
    }

    analyzeTimingQuality() {
        // Analyze overall timing quality
        const avgConfidence = this.segments.reduce((sum, seg) => sum + seg.confidence, 0) / this.segments.length;
        const gapIssues = this.segments.filter(seg => Math.abs(seg.nextSegmentGap) > 0.5).length;

        let quality = 'Good';
        let color = '#9BF04C';

        if (avgConfidence < 0.6 || gapIssues > this.segments.length * 0.1) {
            quality = 'Fair';
            color = '#FBFE65';
        }
        if (avgConfidence < 0.4 || gapIssues > this.segments.length * 0.2) {
            quality = 'Poor';
            color = '#A31ACB';
        }

        const qualityIndicator = document.getElementById('timingQuality');
        if (qualityIndicator) {
            qualityIndicator.textContent = `Sync Quality: ${quality}`;
            qualityIndicator.style.color = color;
        }
    }

    showTooltip(element, text) {
        // Simple tooltip implementation
        const tooltip = document.createElement('div');
        tooltip.className = 'timing-tooltip';
        tooltip.textContent = text;

        const rect = element.getBoundingClientRect();
        tooltip.style.position = 'fixed';
        tooltip.style.left = `${rect.left}px`;
        tooltip.style.top = `${rect.top - 30}px`;

        document.body.appendChild(tooltip);

        element.addEventListener('mouseleave', () => {
            tooltip.remove();
        }, { once: true });
    }

    setupEventListeners() {
        // Audio element event listeners
        this.audio.addEventListener('timeupdate', () => {
            if (!this.isUserSeeking) {
                this.updateHighlight();
            }
        });

        this.audio.addEventListener('seeking', () => {
            this.isUserSeeking = true;
        });

        this.audio.addEventListener('seeked', () => {
            this.isUserSeeking = false;
            this.updateHighlight();
        });

        this.audio.addEventListener('loadedmetadata', () => {
            this.updateTimeDisplay();
        });

        // Control button listeners
        if (this.toggleAutoScrollBtn) {
            this.toggleAutoScrollBtn.addEventListener('click', () => {
                this.autoScroll = !this.autoScroll;
                this.toggleAutoScrollBtn.textContent = `AUTO-SCROLL: ${this.autoScroll ? 'ON' : 'OFF'}`;
            });
        }

        if (this.resumeTrackingBtn) {
            this.resumeTrackingBtn.addEventListener('click', () => {
                this.isUserSeeking = false;
                this.resumeTrackingBtn.style.display = 'none';
                this.updateHighlight();
            });
        }
    }

    seekToTime(time) {
        this.audio.currentTime = time;
        this.isUserSeeking = false;
        this.updateHighlight();
    }

    updateTimeDisplay() {
        if (this.currentTimeDisplay) {
            const current = this.audio.currentTime || 0;
            this.currentTimeDisplay.textContent = this.formatTime(current);
        }

        if (this.totalTimeDisplay && this.audio.duration) {
            this.totalTimeDisplay.textContent = this.formatTime(this.audio.duration);
        }
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showError(message) {
        console.error('Read-along error:', message);
        this.textContainer.innerHTML = `
            <p style="color: #FBFE65; text-align: center; padding: 20px;">
                ❌ ERROR LOADING TEXT SYNCHRONIZATION<br>
                <span style="font-size: 0.9em; color: #50C8FC;">${this.escapeHtml(message)}</span><br>
                <br>
                Check browser console for details.
            </p>
        `;
    }

    destroy() {
        // Cleanup
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }
}

// Enhanced CSS for smooth highlighting
const enhancedStyles = `
<style>
.text-segment {
    position: relative;
    transition: all 0.2s ease;
}

.text-segment.active {
    background: linear-gradient(
        to right,
        #9BF04C 0%,
        #9BF04C calc(var(--highlight-progress, 0) * 100%),
        rgba(155, 240, 76, 0.3) calc(var(--highlight-progress, 0) * 100%),
        rgba(155, 240, 76, 0.3) 100%
    );
}

.text-segment.pre-active {
    animation: pulse 0.5s ease-in-out;
}

.text-segment.upcoming {
    opacity: 0.8;
    border-bottom: 1px dashed rgba(155, 240, 76, 0.5);
}

.text-segment.low-confidence {
    border-bottom: 1px dotted #FBFE65;
}

.segment-type-technical {
    font-family: 'Courier New', monospace;
    background-color: rgba(64, 49, 141, 0.2);
}

.segment-type-emphasis {
    font-weight: bold;
}

.timing-quality-indicator {
    position: sticky;
    top: 0;
    background: rgba(0, 0, 0, 0.9);
    padding: 5px;
    text-align: center;
    font-size: 0.9em;
    z-index: 10;
}

.timing-tooltip {
    background: rgba(0, 0, 0, 0.9);
    color: #50C8FC;
    padding: 5px 10px;
    border-radius: 3px;
    font-size: 0.8em;
    pointer-events: none;
    z-index: 1000;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}
</style>
`;

// Inject styles
document.head.insertAdjacentHTML('beforeend', enhancedStyles);
