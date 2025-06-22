import cv2
import numpy as np
from deepface import DeepFace
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import queue
from datetime import datetime
import os
import json

class InMemoryFaceDatabase:
    def __init__(self):
        self.known_faces = {}  # {person_id: {'encoding': embedding, 'name': name, 'info': {...}}}
        self.next_person_id = 1
        
    def add_face(self, embedding, face_info, custom_name=None):
        """Add new face to in-memory database"""
        if custom_name:
            person_id = custom_name
        else:
            person_id = f"Person_{self.next_person_id}"
            
        self.known_faces[person_id] = {
            'encoding': embedding,
            'name': person_id,
            'age': face_info.get('age', 'Unknown'),
            'gender': face_info.get('dominant_gender', 'Unknown')
        }
        
        if not custom_name:
            self.next_person_id += 1
            
        return person_id
        
    def find_match(self, embedding, threshold=0.6):
        """Find matching face in memory database"""
        if not self.known_faces or embedding is None:
            return None, None
            
        best_match = None
        best_distance = float('inf')
        
        for person_id, person_data in self.known_faces.items():
            try:
                known_encoding = person_data.get('encoding')
                if known_encoding is None:
                    continue
                    
                known_encoding = np.array(known_encoding)
                current_encoding = np.array(embedding)
                
                if known_encoding.size == 0 or current_encoding.size == 0:
                    continue
                
                # Fast cosine distance calculation
                known_norm = np.linalg.norm(known_encoding)
                current_norm = np.linalg.norm(current_encoding)
                
                if known_norm == 0 or current_norm == 0:
                    continue
                
                # Optimized cosine distance
                dot_product = np.dot(known_encoding, current_encoding)
                distance = 1 - (dot_product / (known_norm * current_norm))
                
                if distance < threshold and distance < best_distance:
                    best_distance = distance
                    best_match = person_id
                    
            except Exception:
                continue
                
        if best_match:
            return best_match, best_distance
            
        return None, None
        
    def get_stats(self):
        """Get database statistics"""
        return {
            'total_faces': len(self.known_faces)
        }

class FastFaceRecognizer:
    def __init__(self):
        self.faces = []
        self.face_database = InMemoryFaceDatabase()
        self.frame_queue = queue.Queue(maxsize=1)  # Smaller queue for better performance
        self.result_cache = {}  # Cache recent results
        self.cache_timeout = 1.0  # Cache results for 1 second
        
        # Single background thread for processing
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.running = True
        self.processing_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.processing_thread.start()
        
    def _background_processor(self):
        """Optimized background processing"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Submit processing task
                future = self.executor.submit(self._process_frame, frame)
                
                try:
                    result = future.result(timeout=1.5)
                    if result:
                        self.faces = result
                except:
                    pass  # Keep previous results
                    
            except queue.Empty:
                continue
            except Exception:
                continue
                
    def _process_frame(self, frame):
        """Fast frame processing with caching"""
        try:
            # Aggressive downscaling for speed
            height, width = frame.shape[:2]
            scale_factor = 0.4  # Even smaller for speed
            small_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
            
            # Fast face detection only
            results = DeepFace.analyze(
                small_frame,
                actions=['age', 'gender', 'emotion'],
                detector_backend='opencv',
                enforce_detection=False,
                silent=True
            )
            
            if not isinstance(results, list):
                results = [results] if results else []
            
            processed_faces = []
            
            for result in results:
                if 'region' not in result:
                    continue
                    
                try:
                    # Scale coordinates back
                    region = result['region'].copy()
                    region['x'] = int(region['x'] / scale_factor)
                    region['y'] = int(region['y'] / scale_factor)
                    region['w'] = int(region['w'] / scale_factor)
                    region['h'] = int(region['h'] / scale_factor)
                    result['region'] = region
                    
                    # Extract face region
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    
                    # Skip very small faces for performance
                    if w < 50 or h < 50:
                        continue
                        
                    face_img = frame[max(0,y):y+h, max(0,x):x+w]
                    # Reject if face_img is nearly the size of the whole frame (likely a false detection)
                    frame_h, frame_w = frame.shape[:2]
                    face_h, face_w = face_img.shape[:2]
                    if face_h > 0.9 * frame_h and face_w > 0.9 * frame_w:
                        continue

                    if face_img.size > 0:
                        # Get embedding for recognition
                        embedding = self._get_fast_embedding(face_img)
                        if embedding is not None:
                            # Try to match
                            match_id, distance = self.face_database.find_match(embedding)
                            
                            if match_id and distance is not None:
                                # Known face
                                person_data = self.face_database.known_faces[match_id]
                                result.update({
                                    'person_id': match_id,
                                    'person_name': person_data['name'],
                                    'confidence': f"{(1-distance)*100:.1f}%",
                                    'stored_age': person_data['age'],
                                    'stored_gender': person_data['gender']
                                })
                            else:
                                # New face
                                new_id = self.face_database.add_face(embedding, result)
                                result.update({
                                    'person_id': new_id,
                                    'person_name': new_id,
                                    'confidence': "New",
                                    'stored_age': result.get('age', '?'),
                                    'stored_gender': result.get('dominant_gender', '?')
                                })
                            
                            processed_faces.append(result)
                            
                except Exception:
                    continue
                        
            return processed_faces
            
        except Exception:
            return []
    
    def _get_fast_embedding(self, face_img):
        """Fast embedding extraction with minimal processing"""
        try:
            # For loading from files, we might need to detect face first
            if face_img.shape[0] > 200 or face_img.shape[1] > 200:
                # This is likely a full image, try to detect face
                try:
                    results = DeepFace.analyze(
                        face_img,
                        actions=[],
                        detector_backend='opencv',
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if isinstance(results, list) and results:
                        result = results[0]
                    elif results:
                        result = results
                    else:
                        result = None
                        
                    if result and 'region' in result:
                        region = result['region']
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        face_img = face_img[y:y+h, x:x+w]
                except:
                    pass  # Use the full image if face detection fails
            
            # Resize face for faster embedding
            face_resized = cv2.resize(face_img, (112, 112))  # Standard face size
            
            embedding_result = DeepFace.represent(
                face_resized,
                model_name='Facenet',
                enforce_detection=False
            )
            
            if embedding_result and isinstance(embedding_result, list) and len(embedding_result) > 0:
                return embedding_result[0]['embedding']
                
        except Exception:
            pass
            
        return None
    
    def add_frame(self, frame):
        """Add frame for processing (non-blocking)"""
        try:
            # Clear queue and add new frame
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
                    
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    def stop(self):
        """Stop processing"""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        self.executor.shutdown(wait=False)

def draw_text_with_background(frame, text, position, font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 0), thickness=2):
    """Draw text with background for better visibility"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    # Draw background rectangle
    cv2.rectangle(frame, 
                  (x - 5, y - text_height - 5), 
                  (x + text_width + 5, y + baseline + 5), 
                  bg_color, -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

def draw_face_info(frame, face, face_index):
    """Enhanced drawing function showing person name and emotion"""
    try:
        region = face.get('region', {})
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
        
        if w <= 0 or h <= 0:
            return
            
        # Get info
        person_name = face.get('person_name', 'Unknown')
        current_emotion = face.get('dominant_emotion', 'Unknown')
        confidence = face.get('confidence', 'N/A')
        
        # Color scheme based on recognition status
        if confidence == "New":
            color = (0, 255, 255)  # Yellow for new faces
        else:
            color = (0, 255, 0)  # Green for recognized faces
        
        # Draw nice rounded rectangle (simulate with thick border)
        border_thickness = 3
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, border_thickness)
        
        # Draw corner markers for a modern look
        corner_length = 20
        corner_thickness = 4
        
        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_length, y), color, corner_thickness)
        cv2.line(frame, (x, y), (x, y + corner_length), color, corner_thickness)
        
        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, corner_thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, corner_thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, corner_thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, corner_thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, corner_thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, corner_thickness)
        
        # Display person name and emotion
        name_text = f"{person_name}"
        emotion_text = f"{current_emotion.title()}"
        
        text_x = x
        text_y = y - 40 if y > 50 else y + h + 25
        
        # Draw name
        draw_text_with_background(frame, name_text, (text_x, text_y), 
                                font_scale=0.8, color=color, bg_color=(0, 0, 0, 180))
        
        # Draw emotion below name
        draw_text_with_background(frame, emotion_text, (text_x, text_y + 25), 
                                font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 0, 180))
                           
    except Exception:
        pass

def main():
    print("=" * 50)
    # Initialize recognizer
    recognizer = FastFaceRecognizer()

    # Load embeddings from face_embeddings.json
    embedding_file = "face_embeddings.json"
    if os.path.exists(embedding_file):
        with open(embedding_file, "r") as f:
            data = json.load(f)
        for name in ["Jesse", "Nikita", "Austin"]:
            person = data.get(name)
            if person and "embedding" in person:
                embedding = person["embedding"]
                # Provide dummy info for age/gender if not present
                face_info = {
                    "age": person.get("age", "Unknown"),
                    "dominant_gender": person.get("gender", "Unknown")
                }
                recognizer.face_database.add_face(embedding, face_info, custom_name=name)


    # Initialize camera with performance settings
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Performance optimized settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Performance tracking
    fps_counter = 0
    fps_start = time.time()
    fps = 0
    
    print("\n😊 Face Recognition with Emotion Detection")
    print("⚡ Optimized for real-time performance")
    print("Press 'q' to quit, 'c' to clear memory")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add frame for processing
            recognizer.add_frame(frame)
            
            # Draw faces
            for i, face in enumerate(recognizer.faces):
                draw_face_info(frame, face, i)
            
            # Update FPS
            fps_counter += 1
            if fps_counter % 15 == 0:
                fps = 15 / (time.time() - fps_start)
                fps_start = time.time()
            
            # Display stats
            stats = recognizer.face_database.get_stats()
            draw_text_with_background(frame, f"Known Faces: {stats['total_faces']}", (10, 30), 
                                    font_scale=0.6, color=(255, 255, 255), bg_color=(0, 0, 0))
            draw_text_with_background(frame, f"Detected: {len(recognizer.faces)}", (10, 60), 
                                    font_scale=0.6, color=(255, 255, 255), bg_color=(0, 0, 0))
            
            cv2.imshow('Face Recognition - Emotion Detection', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear memory database
                recognizer.face_database = InMemoryFaceDatabase()
                print("🗑️ Memory cleared!")
                
    except KeyboardInterrupt:
        print("\n👋 Session ended")
    finally:
        recognizer.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("🔥 All data cleared from memory")

if __name__ == "__main__":
    main()