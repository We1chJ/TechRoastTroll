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
import subprocess

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
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_cache = {}
        self.cache_timeout = 1.0
        self.last_process_time = 0
        self.process_interval = 0.1  # Process every 100ms instead of every frame
        
        # Initialize face detector once for reuse
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Single background thread for processing
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.running = True
        self.processing_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.processing_thread.start()
        
    def _background_processor(self):
        """Optimized background processing with frame skipping"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Skip processing if too soon since last process
                current_time = time.time()
                if current_time - self.last_process_time < self.process_interval:
                    continue
                
                # Submit processing task
                future = self.executor.submit(self._process_frame, frame)
                
                try:
                    result = future.result(timeout=1.0)  # Reduced timeout
                    if result is not None:
                        self.faces = result
                        self.last_process_time = current_time
                except:
                    pass  # Keep previous results
                    
            except queue.Empty:
                continue
            except Exception:
                continue
                
    def _process_frame(self, frame):
        """Optimized frame processing with better face detection"""
        try:
            # More aggressive downscaling for initial detection
            height, width = frame.shape[:2]
            scale_factor = 0.3  # Smaller for faster processing
            small_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
            
            # Convert to grayscale for faster OpenCV detection
            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Use OpenCV for fast face detection first
            faces_cv = self.face_cascade.detectMultiScale(
                gray_small,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20),  # Minimum face size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces_cv) == 0:
                return []
            
            processed_faces = []
            
            # Process only the first few faces for performance
            for i, (x, y, w, h) in enumerate(faces_cv[:3]):  # Limit to 3 faces max
                try:
                    # Scale coordinates back to original frame
                    x_orig = int(x / scale_factor)
                    y_orig = int(y / scale_factor)
                    w_orig = int(w / scale_factor)
                    h_orig = int(h / scale_factor)
                    
                    # Add padding for better face extraction
                    padding = 10
                    x_padded = max(0, x_orig - padding)
                    y_padded = max(0, y_orig - padding)
                    w_padded = min(width - x_padded, w_orig + 2*padding)
                    h_padded = min(height - y_padded, h_orig + 2*padding)
                    
                    # Extract face region
                    face_img = frame[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
                    
                    if face_img.size == 0:
                        continue
                    
                    # Get analysis from DeepFace (less frequent)
                    try:
                        # Resize face for faster analysis
                        face_resized = cv2.resize(face_img, (160, 160))
                        
                        analysis = DeepFace.analyze(
                            face_resized,
                            actions=['age', 'gender', 'emotion'],
                            detector_backend='skip',  # Skip detection since we already have the face
                            enforce_detection=False,
                            silent=True
                        )
                        
                        if isinstance(analysis, list):
                            analysis = analysis[0] if analysis else {}
                        
                    except Exception:
                        # Use default values if analysis fails
                        analysis = {
                            'age': 'Unknown',
                            'dominant_gender': 'Unknown', 
                            'dominant_emotion': 'neutral'
                        }
                    
                    # Get embedding for recognition (less frequent)
                    embedding = None
                    if i < 2:  # Only get embeddings for first 2 faces
                        embedding = self._get_fast_embedding(face_img)
                    
                    # Create result object
                    result = {
                        'region': {
                            'x': x_orig,
                            'y': y_orig, 
                            'w': w_orig,
                            'h': h_orig
                        },
                        'age': analysis.get('age', 'Unknown'),
                        'dominant_gender': analysis.get('dominant_gender', 'Unknown'),
                        'dominant_emotion': analysis.get('dominant_emotion', 'neutral')
                    }
                    
                    if embedding is not None:
                        # Try to match with known faces
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
                            # New face - but don't add every frame
                            if time.time() % 3 < 0.5:  # Only add new faces occasionally
                                new_id = self.face_database.add_face(embedding, analysis)
                                result.update({
                                    'person_id': new_id,
                                    'person_name': new_id,
                                    'confidence': "New",
                                    'stored_age': analysis.get('age', '?'),
                                    'stored_gender': analysis.get('dominant_gender', '?')
                                })
                            else:
                                result.update({
                                    'person_id': 'Unknown',
                                    'person_name': 'Unknown',
                                    'confidence': "Processing...",
                                    'stored_age': '?',
                                    'stored_gender': '?'
                                })
                    else:
                        result.update({
                            'person_id': 'Detected',
                            'person_name': 'Face',
                            'confidence': "Detected",
                            'stored_age': '?',
                            'stored_gender': '?'
                        })
                    
                    processed_faces.append(result)
                    
                except Exception:
                    continue
                        
            return processed_faces
            
        except Exception:
            return []
    
    def _get_fast_embedding(self, face_img):
        """Optimized embedding extraction"""
        try:
            # Resize to standard size for consistency
            face_resized = cv2.resize(face_img, (112, 112))
            
            embedding_result = DeepFace.represent(
                face_resized,
                model_name='Facenet',  # Facenet is generally faster than ArcFace
                enforce_detection=False,
                detector_backend='skip'  # Skip detection since we already have the face
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
        elif confidence == "Processing...":
            color = (255, 165, 0)  # Orange for processing
        else:
            color = (0, 255, 0)  # Green for recognized faces
        
        # Draw nice rounded rectangle (simulate with thick border)
        border_thickness = 2  # Thinner for better performance
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, border_thickness)
        
        # Simplified corner markers
        corner_length = 15
        corner_thickness = 3
        
        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_length, y), color, corner_thickness)
        cv2.line(frame, (x, y), (x, y + corner_length), color, corner_thickness)
        
        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, corner_thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, corner_thickness)
        
        # Display person name and emotion
        name_text = f"{person_name}"
        emotion_text = f"{current_emotion.title()}"
        
        text_x = x
        text_y = y - 35 if y > 40 else y + h + 20
        
        # Draw name
        draw_text_with_background(frame, name_text, (text_x, text_y), 
                                font_scale=0.7, color=color, bg_color=(0, 0, 0, 180))
        
        # Draw emotion below name
        draw_text_with_background(frame, emotion_text, (text_x, text_y + 20), 
                                font_scale=0.6, color=(255, 255, 255), bg_color=(0, 0, 0, 180))
                           
    except Exception:
        pass

class TwitchStreamer:
    def __init__(self, width=640, height=480, fps=30, stream_key="your_stream_key_here"):
        self.running = True
        self.width = width
        self.height = height
        self.fps = fps
        self.stream_key = stream_key
        self.rtmp_url = f"rtmp://live.twitch.tv/app/{stream_key}"

        # FFmpeg command to stream
        self.ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',  # Input from stdin
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'veryfast',
            '-f', 'flv',
            self.rtmp_url
        ]

        # Start FFmpeg process
        self.process = subprocess.Popen(self.ffmpeg_cmd, stdin=subprocess.PIPE)

        # Queue for thread-safe streaming
        self.frame_queue = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()

    def _stream_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                if frame is not None:
                    self.process.stdin.write(frame.tobytes())
            except queue.Empty:
                continue
            except Exception:
                break

    def send_frame(self, frame):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put_nowait(frame)

    def stop(self):
        self.running = False
        try:
            if self.thread.is_alive():
                self.thread.join(timeout=1.0)
        except:
            pass
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()


def main():
    print("=" * 50)
    # Initialize recognizer
    recognizer = FastFaceRecognizer()

    # Initialize Twitch streamer
    twitch_streamer = TwitchStreamer(stream_key="live_1249689599_Kmi5FRgzNObJ35wA4i2W5zaDnIFsqv")

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

    # Initialize camera with optimized settings
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Optimized camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Reduce auto-exposure lag
    
    # Performance tracking
    fps_counter = 0
    fps_start = time.time()
    fps = 0
    frame_skip = 0
    
    print("\n😊 Face Recognition with Emotion Detection")
    print("⚡ Optimized for real-time performance")
    print("Press 'q' to quit, 'c' to clear memory")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip every other frame for better performance
            frame_skip += 1
            if frame_skip % 2 == 0:
                recognizer.add_frame(frame)
            
            # Draw faces
            for i, face in enumerate(recognizer.faces):
                draw_face_info(frame, face, i)
            
            # Update FPS less frequently
            fps_counter += 1
            if fps_counter % 30 == 0:  # Update every 30 frames
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
            
            # Display stats
            stats = recognizer.face_database.get_stats()
            draw_text_with_background(frame, f"Known: {stats['total_faces']} | Detected: {len(recognizer.faces)} | FPS: {fps:.1f}", 
                                    (10, 30), font_scale=0.6, color=(255, 255, 255), bg_color=(0, 0, 0))
            
            twitch_streamer.send_frame(frame.copy())
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
        twitch_streamer.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("🔥 All data cleared from memory")

if __name__ == "__main__":
    main()