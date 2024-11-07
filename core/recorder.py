import threading
from typing import Dict, Optional, List, Any
import time
import json
from pathlib import Path
from .element_detector import ElementDetectorManager
from .interaction_tracker import InteractionTracker
from ..utils.config import Config
from ..utils.metrics import MetricsTracker

class Recorder:
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
        self.element_detector = ElementDetectorManager(
            confidence_threshold=self.config.model_config.confidence_threshold
        )
        self.interaction_tracker = InteractionTracker(
            buffer_size=self.config.recorder_config.buffer_size
        )
        self.metrics = MetricsTracker()
        
        self.recording = False
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        
        # Create save directory if it doesn't exist
        Path(self.config.recorder_config.save_directory).mkdir(parents=True, exist_ok=True)
    
    def start_recording(self) -> Dict[str, Any]:
        """Start recording user interactions"""
        if self.recording:
            return {'status': 'error', 'message': 'Recording already in progress'}
        
        try:
            self.recording = True
            self.stop_flag.clear()
            self.metrics.clear()
            self.interaction_tracker.start_recording()
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._process_interactions
            )
            self.processing_thread.start()
            
            return {
                'status': 'success',
                'message': 'Recording started successfully',
                'timestamp': time.time()
            }
        except Exception as e:
            self.recording = False
            return {
                'status': 'error',
                'message': f'Failed to start recording: {str(e)}'
            }
    
    def stop_recording(self) -> Dict[str, Any]:
        """Stop recording user interactions"""
        if not self.recording:
            return {'status': 'error', 'message': 'No recording in progress'}
        
        try:
            self.recording = False
            self.stop_flag.set()
            
            if self.processing_thread:
                self.processing_thread.join(timeout=5.0)
            
            self.interaction_tracker.stop_recording()
            
            return {
                'status': 'success',
                'message': 'Recording stopped successfully',
                'timestamp': time.time(),
                'metrics': self.metrics.get_summary()
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to stop recording: {str(e)}'
            }
    
    def _process_interactions(self) -> None:
        """Background thread for processing interactions"""
        while not self.stop_flag.is_set():
            time.sleep(self.config.recorder_config.capture_interval_ms / 1000)
            
            if not self.recording:
                break
    
    def process_dom_interaction(self, 
                              screenshot: str,
                              dom_string: str,
                              event_type: str,
                              coordinates: Optional[tuple[int, int]] = None,
                              value: Optional[str] = None) -> Dict[str, Any]:
        """Process a DOM-based interaction"""
        if not self.recording:
            return {'status': 'error', 'message': 'Not recording'}
        
        try:
            self.metrics.start_operation()
            
            # Detect elements
            detection_result = self.element_detector.detect_elements_dom(
                screenshot, dom_string
            )
            
            if 'error' in detection_result:
                return {'status': 'error', 'message': detection_result['error']}
            
            # Track interaction
            self.interaction_tracker.track_interaction(
                interaction_type=event_type,
                coordinates=coordinates,
                element_type='dom',
                value=value,
                confidence=detection_result['detection']['confidence']
            )
            
            latency = self.metrics.end_operation()
            
            return {
                'status': 'success',
                'detection': detection_result['detection'],
                'latency_ms': latency
            }
            
        except Exception as e:
            self.metrics.end_operation()
            return {'status': 'error', 'message': str(e)}
    
    def process_canvas_interaction(self,
                                 screenshot: str,
                                 event_type: str,
                                 coordinates: Optional[tuple[int, int]] = None,
                                 value: Optional[str] = None) -> Dict[str, Any]:
        """Process a canvas-based interaction"""
        if not self.recording:
            return {'status': 'error', 'message': 'Not recording'}
        
        try:
            self.metrics.start_operation()
            
            # Detect elements
            detection_result = self.element_detector.detect_elements_canvas(
                screenshot
            )
            
            if 'error' in detection_result:
                return {'status': 'error', 'message': detection_result['error']}
            
            # Track interaction
            self.interaction_tracker.track_interaction(
                interaction_type=event_type,
                coordinates=coordinates,
                element_type='canvas',
                value=value,
                confidence=detection_result['detection']['confidence']
            )
            
            latency = self.metrics.end_operation()
            
            return {
                'status': 'success',
                'detection': detection_result['detection'],
                'latency_ms': latency
            }
            
        except Exception as e:
            self.metrics.end_operation()
            return {'status': 'error', 'message': str(e)}
    
    def save_recording(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Save the current recording to a file"""
        try:
            if filename is None:
                filename = f"recording_{int(time.time())}.json"
            
            filepath = Path(self.config.recorder_config.save_directory) / filename
            
            # Save interactions
            self.interaction_tracker.save_recording(str(filepath))
            
            # Save metrics
            metrics_filepath = filepath.with_suffix('.metrics.json')
            with open(metrics_filepath, 'w') as f:
                json.dump(self.metrics.get_summary(), f, indent=2)
            
            return {
                'status': 'success',
                'message': 'Recording saved successfully',
                'filepath': str(filepath),
                'metrics_filepath': str(metrics_filepath)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to save recording: {str(e)}'
            }
    
    def load_recording(self, filepath: str) -> Dict[str, Any]:
        """Load a recording from a file"""
        try:
            self.interaction_tracker.load_recording(filepath)
            
            # Try to load metrics if they exist
            metrics_filepath = Path(filepath).with_suffix('.metrics.json')
            if metrics_filepath.exists():
                with open(metrics_filepath, 'r') as f:
                    metrics_data = json.load(f)
                    self.metrics = MetricsTracker()
                    # Populate metrics (simplified)
                    self.metrics.metrics.latencies = metrics_data.get('latency', {}).get('all', [])
            
            return {
                'status': 'success',
                'message': 'Recording loaded successfully',
                'interaction_count': len(self.interaction_tracker.get_interactions())
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to load recording: {str(e)}'
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'metrics': self.metrics.get_summary(),
            'interaction_count': len(self.interaction_tracker.get_interactions())
        }
    
    def clear(self) -> Dict[str, str]:
        """Clear all recorded data and metrics"""
        try:
            self.interaction_tracker.buffer.clear()
            self.metrics.clear()
            return {
                'status': 'success',
                'message': 'Recorder data cleared successfully'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to clear recorder data: {str(e)}'
            }