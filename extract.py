import cv2
import numpy as np
from deepface import DeepFace
import os
import json

class FaceEmbeddingExtractor:
    def __init__(self):
        pass
    
    def extract_embedding(self, image_path):
        """
        Extract face embedding from a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: {
                'success': bool,
                'embedding': list or None,
                'face_info': dict or None,
                'error': str or None
            }
        """
        try:
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'embedding': None,
                    'face_info': None,
                    'error': f"Image not found: {image_path}"
                }
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {
                    'success': False,
                    'embedding': None,
                    'face_info': None,
                    'error': f"Could not load image: {image_path}"
                }
            
            # Get face info (age, gender)
            face_info = None
            try:
                results = DeepFace.analyze(
                    img,
                    actions=['age', 'gender'],
                    detector_backend='opencv',
                    enforce_detection=False,
                    silent=True
                )
                
                if isinstance(results, list) and results:
                    face_info = results[0]
                elif results:
                    face_info = results
                    
            except Exception:
                face_info = {'age': 'Unknown', 'dominant_gender': 'Unknown'}
            
            # Extract embedding
            embedding = self._get_embedding(img)
            
            if embedding is None:
                return {
                    'success': False,
                    'embedding': None,
                    'face_info': face_info,
                    'error': "Could not extract embedding"
                }
            
            return {
                'success': True,
                'embedding': embedding,
                'face_info': face_info,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'embedding': None,
                'face_info': None,
                'error': str(e)
            }
    
    def _get_embedding(self, img):
        """Extract embedding from image"""
        try:
            # Try to detect face first for full images
            if img.shape[0] > 200 or img.shape[1] > 200:
                try:
                    results = DeepFace.analyze(
                        img,
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
                        img = img[y:y+h, x:x+w]
                except:
                    pass  # Use full image if face detection fails
            
            # Resize for consistent embedding
            img_resized = cv2.resize(img, (112, 112))
            
            # Get embedding using FaceNet
            embedding_result = DeepFace.represent(
                img_resized,
                model_name='Facenet',
                enforce_detection=False
            )
            
            if embedding_result and isinstance(embedding_result, list) and len(embedding_result) > 0:
                return embedding_result[0]['embedding']
                
        except Exception:
            pass
            
        return None
    
    def extract_multiple(self, image_paths_and_names):
        """
        Extract embeddings from multiple images
        
        Args:
            image_paths_and_names: List of tuples [(image_path, person_name), ...]
            
        Returns:
            dict: {person_name: embedding_result}
        """
        results = {}
        
        print("📸 Extracting embeddings from images...")
        
        for image_path, person_name in image_paths_and_names:
            print(f"🔍 Processing {person_name} from {image_path}...")
            
            result = self.extract_embedding(image_path)
            results[person_name] = result
            
            if result['success']:
                print(f"✅ Successfully extracted embedding for {person_name}")
            else:
                print(f"❌ Failed to extract embedding for {person_name}: {result['error']}")
        
        return results
    
    def save_embeddings(self, embeddings_dict, output_file):
        """
        Save embeddings to a JSON file
        
        Args:
            embeddings_dict: Dictionary of embeddings from extract_multiple()
            output_file: Path to save the JSON file
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_dict = {}
            for name, data in embeddings_dict.items():
                if data['success'] and data['embedding'] is not None:
                    serializable_dict[name] = {
                        'embedding': data['embedding'] if isinstance(data['embedding'], list) else data['embedding'].tolist(),
                        'face_info': data['face_info']
                    }
            
            with open(output_file, 'w') as f:
                json.dump(serializable_dict, f, indent=2)
                
            print(f"💾 Embeddings saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error saving embeddings: {str(e)}")
    
    def load_embeddings(self, input_file):
        """
        Load embeddings from a JSON file
        
        Args:
            input_file: Path to the JSON file
            
        Returns:
            dict: Dictionary of embeddings
        """
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            print(f"📂 Embeddings loaded from {input_file}")
            return data
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {str(e)}")
            return {}

def main():
    """Example usage"""
    extractor = FaceEmbeddingExtractor()
    
    # # Example 1: Single image
    # print("🔬 Single Image Example:")
    # result = extractor.extract_embedding("./faces/jesse.png")
    # if result['success']:
    #     print(f"✅ Embedding extracted! Shape: {len(result['embedding'])}")
    #     print(f"Face info: {result['face_info']}")
    # else:
    #     print(f"❌ Error: {result['error']}")
    
    # print("\n" + "="*50 + "\n")
    
    # Example 2: Multiple images
    print("🔬 Multiple Images Example:")
    people_to_process = [
        ("./faces/jesse.png", "Jesse"),
        ("./faces/nikita.png", "Nikita"), 
        ("./faces/austin.png", "Austin"),
    ]
    
    results = extractor.extract_multiple(people_to_process)
    
    # Save embeddings
    extractor.save_embeddings(results, "face_embeddings.json")
    
    # Load embeddings (example)
    loaded_embeddings = extractor.load_embeddings("face_embeddings.json")
    print(f"📊 Loaded {len(loaded_embeddings)} face embeddings")

if __name__ == "__main__":
    main()