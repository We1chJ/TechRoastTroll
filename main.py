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
from dotenv import load_dotenv
import random

JOB_TITLES = [
    "Software Engineer",
    "Senior Software Engineer",
    "Principal Engineer",
    "Engineering Manager",
    "Product Manager",
    "Data Scientist",
    "Machine Learning Engineer",
    "DevOps Engineer",
    "Cloud Architect",
    "Site Reliability Engineer",
    "QA Engineer",
    "Frontend Developer",
    "Backend Developer",
    "Full Stack Developer",
    "Mobile Developer",
    "Security Engineer",
    "UI/UX Designer",
    "Solutions Architect",
    "Technical Program Manager",
    "Research Scientist",
    "Database Administrator",
    "Network Engineer",
    "Support Engineer",
    "Systems Engineer",
    "Hardware Engineer",
    "Embedded Systems Engineer",
    "Game Developer",
    "AI Researcher",
    "Business Analyst",
    "Scrum Master",
    "Release Manager"
]

COMPANIES = [
    "Google",
    "Apple",
    "Microsoft",
    "Amazon",
    "Meta",
    "Netflix",
    "Tesla",
    "OpenAI",
    "Anthropic",
    "SpaceX",
    "Nvidia",
    "Adobe",
    "Salesforce",
    "Oracle",
    "IBM",
    "Intel",
    "AMD",
    "Uber",
    "Airbnb",
    "Stripe",
    "Shopify",
    "Zoom",
    "Slack",
    "Discord",
    "Reddit",
    "Twitter",
    "LinkedIn",
    "GitHub",
    "Atlassian",
    "Figma",
    "Notion",
    "Dropbox",
    "Box",
    "Palantir",
    "Snowflake",
    "Datadog",
    "MongoDB",
    "Twilio",
    "Square",
    "PayPal",
    "Coinbase",
    "Robinhood",
    "DoorDash",
    "Instacart",
    "Lyft",
    "Pinterest",
    "Snapchat",
    "TikTok",
    "Spotify",
    "Unity",
    "Epic Games",
    "Roblox",
    "Cloudflare",
    "Fastly",
    "Vercel",
    "Supabase",
    "Firebase",
    "Heroku",
    "DigitalOcean"
]

def random_job_title():
    return random.choice(JOB_TITLES)

def random_company():
    return random.choice(COMPANIES)

class InMemoryFaceDatabase:
    def __init__(self):
        self.known_faces = {}  # {person_id: {'encoding': embedding, 'name': name, 'info': {...}}}
        self.next_person_id = 1

    def _generate_random_compensation(self):
        """Generates a random salary with a wide distribution."""
        # Use a log-normal distribution for more realistic, wider, and random salaries
        base_salary = int(np.random.lognormal(mean=11.2, sigma=0.45))  # mean/sigma tuned for 50k-400k+
        # Add a random chance for extremely high or low salaries
        if random.random() < 0.05:
            base_salary = int(base_salary * random.uniform(0.3, 0.6))  # rare lowball
        elif random.random() > 0.97:
            base_salary = int(base_salary * random.uniform(1.5, 3.5))  # rare high roller
        bonus = random.randint(5000, 500000)
        total_compensation = base_salary + bonus
        return f"${total_compensation:,}"

    def add_face(self, embedding, face_info, custom_name=None):
        """Add new face to in-memory database"""
        if custom_name:
            person_id = custom_name
        else:
            person_id = f"Person_{self.next_person_id}"

        self.known_faces[person_id] = {
            'encoding': embedding,
            'name': person_id,
            'title': random_job_title(),
            'company': random_company(),  # Add random company assignment
            'compensation': self._generate_random_compensation(), # Add random compensation
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
        self.face_positions = {}  # Track face positions for stability
        self.face_database = InMemoryFaceDatabase()
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_cache = {}
        self.cache_timeout = 3.0  # Keep cached results longer
        self.last_process_time = 0
        self.process_interval = 0.1  # Process every 100ms for stability
        self.position_threshold = 50  # Pixels - faces within this distance are considered same

        # Initialize face detector once for reuse
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Single background thread for processing
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.running = True
        self.processing_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.processing_thread.start()

    def _background_processor(self):
        """Background processing with stable face tracking"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)

                # Process every frame for detection, but do expensive operations less frequently
                current_time = time.time()
                do_full_analysis = current_time - self.last_process_time > self.process_interval

                # Always detect faces for stable boxes
                detected_faces = self._detect_faces_fast(frame)

                if do_full_analysis:
                    # Do full processing less frequently
                    future = self.executor.submit(self._process_frame_full, frame, detected_faces)
                    try:
                        result = future.result(timeout=2.0)
                        if result is not None:
                            self.faces = result
                            self.last_process_time = current_time
                    except:
                        # If full processing fails, keep the basic detection
                        self.faces = detected_faces
                else:
                    # Update positions but keep existing data
                    self.faces = self._update_face_positions(detected_faces, self.faces)

            except queue.Empty:
                continue
            except Exception:
                continue

    def _detect_faces_fast(self, frame):
        """Fast face detection for stable boxes"""
        try:
            height, width = frame.shape[:2]
            scale_factor = 0.5  # Less aggressive downscaling
            small_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            faces_cv = self.face_cascade.detectMultiScale(
                gray_small,
                scaleFactor=1.05,  # More sensitive detection
                minNeighbors=3,    # Lower threshold for more detections
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            detected_faces = []
            for (x, y, w, h) in faces_cv:
                # Scale back to original
                x_orig = int(x / scale_factor)
                y_orig = int(y / scale_factor)
                w_orig = int(w / scale_factor)
                h_orig = int(h / scale_factor)

                detected_faces.append({
                    'region': {
                        'x': x_orig,
                        'y': y_orig,
                        'w': w_orig,
                        'h': h_orig
                    },
                    'person_id': 'Detected',
                    'person_name': 'Processing...',
                    'confidence': "Processing",
                    'age': '?',
                    'dominant_gender': '?',
                    'dominant_emotion': 'processing',
                    'title': '...',
                    'company': '...',
                    'compensation': '...',
                    'processing': True  # Flag to indicate still processing
                })

            return detected_faces

        except Exception:
            return []

    def _update_face_positions(self, detected_faces, existing_faces):
        """Update face positions while maintaining existing data"""
        if not existing_faces:
            return detected_faces

        updated_faces = []

        for detected in detected_faces:
            best_match = None
            best_distance = float('inf')

            detected_center = (
                detected['region']['x'] + detected['region']['w'] // 2,
                detected['region']['y'] + detected['region']['h'] // 2
            )

            # Find closest existing face
            for existing in existing_faces:
                existing_center = (
                    existing['region']['x'] + existing['region']['w'] // 2,
                    existing['region']['y'] + existing['region']['h'] // 2
                )

                distance = ((detected_center[0] - existing_center[0]) ** 2 +
                            (detected_center[1] - existing_center[1]) ** 2) ** 0.5

                if distance < self.position_threshold and distance < best_distance:
                    best_distance = distance
                    best_match = existing

            if best_match:
                # Update position but keep existing data
                updated_face = best_match.copy()
                updated_face['region'] = detected['region']
                updated_faces.append(updated_face)
            else:
                # New face
                updated_faces.append(detected)

        return updated_faces

    def _process_frame_full(self, frame, detected_faces):
        """Full processing for detailed analysis"""
        try:
            processed_faces = []

            for i, face in enumerate(detected_faces[:3]):  # Limit to 3 faces
                try:
                    region = face['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']

                    # Extract face region with padding
                    padding = 10
                    x_padded = max(0, x - padding)
                    y_padded = max(0, y - padding)
                    w_padded = min(frame.shape[1] - x_padded, w + 2*padding)
                    h_padded = min(frame.shape[0] - y_padded, h + 2*padding)

                    face_img = frame[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]

                    if face_img.size == 0:
                        processed_faces.append(face)
                        continue

                    # Get analysis
                    try:
                        face_resized = cv2.resize(face_img, (160, 160))
                        analysis = DeepFace.analyze(
                            face_resized,
                            actions=['age', 'gender', 'emotion'],
                            detector_backend='skip',
                            enforce_detection=False,
                            silent=True
                        )

                        if isinstance(analysis, list):
                            analysis = analysis[0] if analysis else {}

                    except Exception:
                        analysis = {
                            'age': 'Unknown',
                            'dominant_gender': 'Unknown',
                            'dominant_emotion': 'neutral'
                        }

                    # Get embedding for recognition
                    embedding = self._get_fast_embedding(face_img)

                    result = face.copy()
                    result.update({
                        'age': analysis.get('age', 'Unknown'),
                        'dominant_gender': analysis.get('dominant_gender', 'Unknown'),
                        'dominant_emotion': analysis.get('dominant_emotion', 'neutral'),
                        'processing': False  # No longer processing
                    })

                    if embedding is not None:
                        # Try to match with known faces
                        match_id, distance = self.face_database.find_match(embedding)

                        if match_id and distance is not None:
                            person_data = self.face_database.known_faces[match_id]
                            result.update({
                                'person_id': match_id,
                                'person_name': person_data['name'],
                                'confidence': f"{(1-distance)*100:.1f}%",
                                'stored_age': person_data['age'],
                                'stored_gender': person_data['gender'],
                                'title': person_data['title'],
                                'company': person_data['company'],
                                'compensation': person_data['compensation']
                            })
                        else:
                            # New face
                            new_id = self.face_database.add_face(embedding, analysis)
                            person_data = self.face_database.known_faces[new_id]
                            result.update({
                                'person_id': new_id,
                                'person_name': new_id,
                                'confidence': "New",
                                'stored_age': analysis.get('age', '?'),
                                'stored_gender': analysis.get('dominant_gender', '?'),
                                'title': person_data['title'],
                                'company': person_data['company'],
                                'compensation': person_data['compensation']
                            })

                    processed_faces.append(result)

                except Exception:
                    processed_faces.append(face)
                    continue

            return processed_faces

        except Exception:
            return detected_faces

    def _get_fast_embedding(self, face_img):
        """Optimized embedding extraction"""
        try:
            face_resized = cv2.resize(face_img, (112, 112))

            embedding_result = DeepFace.represent(
                face_resized,
                model_name='ArcFace',  # More accurate than Facenet
                enforce_detection=False,
                detector_backend='skip'
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
            self.processing_thread.join(timeout=2.0)
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
    """Enhanced drawing function showing only green bounding boxes with labels only when processed"""
    try:
        region = face.get('region', {})
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)

        if w <= 0 or h <= 0:
            return

        # Always use green color for bounding boxes
        color = (0, 255, 0)  # Green

        # Draw rectangle - always visible and always green
        border_thickness = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, border_thickness)

        # Corner markers
        corner_length = 15
        corner_thickness = 3

        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_length, y), color, corner_thickness)
        cv2.line(frame, (x, y), (x, y + corner_length), color, corner_thickness)

        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, corner_thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, corner_thickness)

        # Only show labels if processing is complete
        is_processing = face.get('processing', True)
        confidence = face.get('confidence', 'N/A')

        if not is_processing and confidence != "Processing":
            # Get info for labels
            person_name = face.get('person_name', 'Unknown')
            current_emotion = face.get('dominant_emotion', 'Unknown')
            title = face.get('title', '???')
            company = face.get('company', '???')
            compensation = face.get('compensation', '???')


            text_x = x
            text_y = y - 50 if y > 60 else y + h + 20

            # Draw labels only when processing is complete
            draw_text_with_background(frame, title, (text_x, text_y + 40),
                                      font_scale=0.55, color=(180, 200, 255), bg_color=(0, 0, 0, 180))

            draw_text_with_background(frame, company, (text_x, text_y + 60),
                                      font_scale=0.55, color=(100, 255, 100), bg_color=(0, 0, 0, 180))

            draw_text_with_background(frame, compensation, (text_x, text_y + 80),
                                      font_scale=0.55, color=(255, 215, 0), bg_color=(0, 0, 0, 180))


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
        self.running = True
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
    load_dotenv(".env.local")
    stream_key = os.environ.get("TWITCH_STREAM_KEY", "YOUR_KEY")
    twitch_streamer = TwitchStreamer(stream_key=stream_key)

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
                # Add the face with custom name (this will assign random company/title)
                recognizer.face_database.add_face(embedding, face_info, custom_name=name)

    # Initialize camera with optimized settings
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Optimized camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    # Performance tracking
    fps_counter = 0
    fps_start = time.time()
    fps = 0

    print("\n😊 Face Recognition with Sticky Green Boxes")
    print("⚡ Green boxes stay visible, labels only after processing")
    print("Press 'q' to quit, 'c' to clear memory")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every frame for stable detection
            recognizer.add_frame(frame)

            # Draw faces - boxes should be stable now
            for i, face in enumerate(recognizer.faces):
                draw_face_info(frame, face, i)

            # Update FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()

            # Display stats
            stats = recognizer.face_database.get_stats()
            draw_text_with_background(frame, f"Known: {stats['total_faces']} | Detected: {len(recognizer.faces)} | FPS: {fps:.1f}",
                                      (10, 30), font_scale=0.6, color=(255, 255, 255), bg_color=(0, 0, 0))

            # twitch_streamer.send_frame(frame.copy())
            cv2.imshow('Face Recognition - Sticky Green Boxes', frame)

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