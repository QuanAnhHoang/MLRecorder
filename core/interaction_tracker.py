from typing import Dict, List, Optional, Union
import time
from dataclasses import dataclass
import json
import queue
import threading

@dataclass
class Interaction:
    timestamp: float
    interaction_type: str
    coordinates: Optional[tuple[int, int]] = None
    element_type: Optional[str] = None
    value: Optional[str] = None
    confidence: float = 1.0

class InteractionBuffer:
    def __init__(self, max_size: int = 1000):
        self.buffer = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()
    
    def add(self, interaction: Interaction) -> None:
        try:
            self.buffer.put_nowait(interaction)
        except queue.Full:
            # Remove oldest interaction if buffer is full
            try:
                self.buffer.get_nowait()
                self.buffer.put_nowait(interaction)
            except queue.Empty:
                pass
    
    def get_all(self) -> List[Interaction]:
        interactions = []
        while not self.buffer.empty():
            try:
                interactions.append(self.buffer.get_nowait())
            except queue.Empty:
                break
        return interactions
    
    def clear(self) -> None:
        with self.lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    break

class InteractionTracker:
    def __init__(self, buffer_size: int = 1000):
        self.buffer = InteractionBuffer(max_size=buffer_size)
        self.recording = False
        self.start_time = None
        
    def start_recording(self) -> None:
        self.recording = True
        self.start_time = time.time()
        self.buffer.clear()
    
    def stop_recording(self) -> None:
        self.recording = False
    
    def track_interaction(self, 
                         interaction_type: str,
                         coordinates: Optional[tuple[int, int]] = None,
                         element_type: Optional[str] = None,
                         value: Optional[str] = None,
                         confidence: float = 1.0) -> None:
        if not self.recording:
            return
            
        interaction = Interaction(
            timestamp=time.time() - self.start_time,
            interaction_type=interaction_type,
            coordinates=coordinates,
            element_type=element_type,
            value=value,
            confidence=confidence
        )
        
        self.buffer.add(interaction)
    
    def get_interactions(self) -> List[Dict]:
        interactions = self.buffer.get_all()
        return [
            {
                'timestamp': interaction.timestamp,
                'type': interaction.interaction_type,
                'coordinates': interaction.coordinates,
                'element_type': interaction.element_type,
                'value': interaction.value,
                'confidence': interaction.confidence
            }
            for interaction in interactions
        ]
    
    def save_recording(self, filepath: str) -> None:
        interactions = self.get_interactions()
        recording_data = {
            'start_time': self.start_time,
            'duration': time.time() - self.start_time if self.start_time else 0,
            'interactions': interactions
        }
        
        with open(filepath, 'w') as f:
            json.dump(recording_data, f, indent=2)
    
    def load_recording(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            recording_data = json.load(f)
        
        self.buffer.clear()
        self.start_time = recording_data['start_time']
        
        for interaction_data in recording_data['interactions']:
            interaction = Interaction(
                timestamp=interaction_data['timestamp'],
                interaction_type=interaction_data['type'],
                coordinates=tuple(interaction_data['coordinates']) if interaction_data['coordinates'] else None,
                element_type=interaction_data['element_type'],
                value=interaction_data['value'],
                confidence=interaction_data['confidence']
            )
            self.buffer.add(interaction)