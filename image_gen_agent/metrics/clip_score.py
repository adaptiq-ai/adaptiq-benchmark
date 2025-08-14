import clip
import torch
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
from pydantic import BaseModel

# Pydantic Models
class CLIPScore(BaseModel):
    raw_score: float
    scaled_score: float
    method_name: str

class ImageComparisonScore(BaseModel):
    raw_score: float
    scaled_score: float

class CLIPAnalysisResult(BaseModel):
    original_image_path: str
    adaptiq_image_path: str
    gpt_image_path: str
    adaptiq_text_prompt: Optional[str]
    gpt_text_prompt: Optional[str]
    adaptiq_text_score: Optional[CLIPScore]
    gpt_text_score: Optional[CLIPScore]
    adaptiq_image_score: ImageComparisonScore
    gpt_image_score: ImageComparisonScore
    raw_difference: float
    scaled_difference: float
    winner: str
    winner_symbol: str
    improvement_percentage: float

class CLIPScoreCalculator:
    """A class to calculate CLIP scores for image-text and image-image comparisons."""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Initialize CLIP model for score calculation
        
        Args:
            model_name: CLIP model variant to use
                       Options: "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", 
                               "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print(f"Loaded CLIP model: {model_name}")
    
    def calculate_text_to_image_score(self, 
                                    image_path: Union[str, Path], 
                                    text_prompt: str,
                                    method_name: str) -> Optional[CLIPScore]:
        """
        Calculate CLIP score between text prompt and image
        
        Args:
            image_path: Path to the image file
            text_prompt: Text description/prompt
            method_name: Name of the method (e.g., "AdaptiQ", "GPT")
            
        Returns:
            CLIPScore: CLIP score results (higher means better alignment)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text
            text_input = clip.tokenize([text_prompt], truncate=True).to(self.device)
            
            # Calculate features
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity (CLIP score)
                clip_score = torch.matmul(image_features, text_features.T).item()
                
                # Convert to more interpretable scale (0-100)
                clip_score_scaled = (clip_score + 1) * 50
                
            return CLIPScore(
                raw_score=clip_score,
                scaled_score=clip_score_scaled,
                method_name=method_name
            )
            
        except Exception as e:
            print(f"Error calculating text-to-image score: {e}")
            return None
    
    def calculate_image_to_image_score(self, 
                                     image1_path: Union[str, Path], 
                                     image2_path: Union[str, Path]) -> Optional[ImageComparisonScore]:
        """
        Calculate CLIP similarity score between two images
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            ImageComparisonScore: CLIP similarity score (higher means more similar)
        """
        try:
            # Load and preprocess images
            image1 = Image.open(image1_path).convert('RGB')
            image2 = Image.open(image2_path).convert('RGB')
            
            image1_input = self.preprocess(image1).unsqueeze(0).to(self.device)
            image2_input = self.preprocess(image2).unsqueeze(0).to(self.device)
            
            # Calculate features
            with torch.no_grad():
                image1_features = self.model.encode_image(image1_input)
                image2_features = self.model.encode_image(image2_input)
                
                # Normalize features
                image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
                image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity
                similarity_score = torch.matmul(image1_features, image2_features.T).item()
                
                # Convert to more interpretable scale (0-100)
                similarity_scaled = (similarity_score + 1) * 50
                
            return ImageComparisonScore(
                raw_score=similarity_score,
                scaled_score=similarity_scaled
            )
            
        except Exception as e:
            print(f"Error calculating image-to-image score: {e}")
            return None
    
    def analyze_clip_score(self, 
                          original_image_path: Union[str, Path],
                          adaptiq_image_path: Union[str, Path], 
                          gpt_image_path: Union[str, Path],
                          adaptiq_text_prompt: Optional[str] = None,
                          gpt_text_prompt: Optional[str] = None) -> CLIPAnalysisResult:
        """
        Comprehensive CLIP analysis comparing AdaptiQ and GPT methods with optional text scoring
        
        Args:
            original_image_path: Path to original/reference image
            adaptiq_image_path: Path to image generated by AdaptiQ method
            gpt_image_path: Path to image generated by GPT method
            adaptiq_text_prompt: Optional text prompt for AdaptiQ text-to-image scoring
            gpt_text_prompt: Optional text prompt for GPT text-to-image scoring
            
        Returns:
            CLIPAnalysisResult: Comprehensive analysis results
        """
        print("🔍 Analyzing CLIP Scores...")
        print(f"Original: {original_image_path}")
        print(f"AdaptiQ: {adaptiq_image_path}")
        print(f"GPT: {gpt_image_path}")
        if adaptiq_text_prompt:
            print(f"AdaptiQ Text Prompt: {adaptiq_text_prompt}")
        if gpt_text_prompt:
            print(f"GPT Text Prompt: {gpt_text_prompt}")
        print("-" * 50)
        
        # Calculate image-to-image scores
        adaptiq_image_score = self.calculate_image_to_image_score(original_image_path, adaptiq_image_path)
        gpt_image_score = self.calculate_image_to_image_score(original_image_path, gpt_image_path)
        
        if adaptiq_image_score is None or gpt_image_score is None:
            print("❌ Failed to calculate image similarity scores")
            return None
        
        # Calculate text-to-image scores if text prompts provided
        adaptiq_text_score = None
        gpt_text_score = None
        if adaptiq_text_prompt:
            adaptiq_text_score = self.calculate_text_to_image_score(adaptiq_image_path, adaptiq_text_prompt, "AdaptiQ")
        if gpt_text_prompt:
            gpt_text_score = self.calculate_text_to_image_score(gpt_image_path, gpt_text_prompt, "GPT")
        
        # Calculate difference based on image similarity scores
        raw_difference = adaptiq_image_score.raw_score - gpt_image_score.raw_score
        scaled_difference = adaptiq_image_score.scaled_score - gpt_image_score.scaled_score
        
        # Determine winner
        if abs(raw_difference) < 0.01:  # Very close scores
            winner = "Tie"
            winner_symbol = "🤝"
        elif raw_difference > 0:
            winner = "AdaptiQ"
            winner_symbol = "🏆"
        else:
            winner = "GPT"
            winner_symbol = "🏆"
        
        result = CLIPAnalysisResult(
            original_image_path=str(original_image_path),
            adaptiq_image_path=str(adaptiq_image_path),
            gpt_image_path=str(gpt_image_path),
            adaptiq_text_prompt=adaptiq_text_prompt,
            gpt_text_prompt=gpt_text_prompt,
            adaptiq_text_score=adaptiq_text_score,
            gpt_text_score=gpt_text_score,
            adaptiq_image_score=adaptiq_image_score,
            gpt_image_score=gpt_image_score,
            raw_difference=raw_difference,
            scaled_difference=scaled_difference,
            winner=winner,
            winner_symbol=winner_symbol,
            improvement_percentage=abs(scaled_difference)
        )
        
        # Print results
        self._print_analysis_results(result)
        
        return result
    
    def _print_analysis_results(self, results: CLIPAnalysisResult):
        """Print formatted analysis results"""
        print("📊 CLIP Analysis Results:")
        print("=" * 50)
        
        # Image Similarity Scores
        print("🖼️ Image Similarity Scores (vs Original):")
        print(f"🔵 AdaptiQ Method:")
        print(f"   Raw Score: {results.adaptiq_image_score.raw_score:.4f}")
        print(f"   Scaled Score: {results.adaptiq_image_score.scaled_score:.2f}/100")
        
        print(f"🟢 GPT Method:")
        print(f"   Raw Score: {results.gpt_image_score.raw_score:.4f}")
        print(f"   Scaled Score: {results.gpt_image_score.scaled_score:.2f}/100")
        
        # Text-to-Image Scores (if available)
        if results.adaptiq_text_prompt and results.adaptiq_text_score:
            print(f"\n📝 AdaptiQ Text-to-Image Alignment:")
            print(f"   Prompt: '{results.adaptiq_text_prompt}'")
            print(f"   Raw Score: {results.adaptiq_text_score.raw_score:.4f}")
            print(f"   Scaled Score: {results.adaptiq_text_score.scaled_score:.2f}/100")
            
        if results.gpt_text_prompt and results.gpt_text_score:
            print(f"\n📝 GPT Text-to-Image Alignment:")
            print(f"   Prompt: '{results.gpt_text_prompt}'")
            print(f"   Raw Score: {results.gpt_text_score.raw_score:.4f}")
            print(f"   Scaled Score: {results.gpt_text_score.scaled_score:.2f}/100")
        
        print("-" * 30)
        
        # Difference Analysis
        print(f"📈 Difference Analysis (Image Similarity):")
        print(f"   Raw Difference: {results.raw_difference:+.4f}")
        print(f"   Scaled Difference: {results.scaled_difference:+.2f} points")
        print(f"   Improvement: {results.improvement_percentage:.2f}%")
        
        print("-" * 30)
        
        # Winner Declaration
        if results.winner == "Tie":
            print(f"{results.winner_symbol} Result: Very close performance - essentially a tie!")
        else:
            print(f"{results.winner_symbol} Winner: {results.winner} method performs better!")
            
        # Performance interpretation
        improvement = results.improvement_percentage
        if improvement >= 5.0:
            print("💪 Significant improvement!")
        elif improvement >= 2.0:
            print("✨ Moderate improvement")
        elif improvement >= 1.0:
            print("👍 Small improvement")
        else:
            print("🤷 Minimal difference")
        
        print("=" * 50)
    
    def batch_text_to_image_scores(self, 
                                  image_paths: List[Union[str, Path]], 
                                  text_prompts: List[str],
                                  method_names: List[str]) -> List[Optional[CLIPScore]]:
        """
        Calculate CLIP scores for multiple image-text pairs
        
        Args:
            image_paths: List of image file paths
            text_prompts: List of text prompts (should match length of image_paths)
            method_names: List of method names (should match length of image_paths)
            
        Returns:
            List[CLIPScore]: List of CLIP score objects
        """
        if len(image_paths) != len(text_prompts) or len(image_paths) != len(method_names):
            raise ValueError("Number of images, text prompts, and method names must match")
        
        scores = []
        for img_path, text, method in zip(image_paths, text_prompts, method_names):
            score = self.calculate_text_to_image_score(img_path, text, method)
            scores.append(score)
            
        return scores
    
    def calculate_average_score(self, scores: List[Optional[CLIPScore]]) -> Optional[Dict[str, float]]:
        """
        Calculate average CLIP score from a list of scores
        
        Args:
            scores: List of CLIP score objects
            
        Returns:
            Dict: Average raw and scaled scores
        """
        valid_scores = [s for s in scores if s is not None]
        
        if not valid_scores:
            return None
            
        avg_raw = np.mean([s.raw_score for s in valid_scores])
        avg_scaled = np.mean([s.scaled_score for s in valid_scores])
        
        return {
            'avg_raw_score': avg_raw,
            'avg_scaled_score': avg_scaled,
            'num_valid_scores': len(valid_scores)
        }
    

