'use server';

import { z } from 'zod';
import { getLabels, getModels, getTestImages } from '@/lib/data';
import type { AllPredictionsResult, PredictionResult, Model, TestImage } from '@/lib/types';
import { summarizePredictionResults } from '@/ai/flows/summarize-prediction-results';

const schema = z.object({
  image: z.instanceof(File).optional(),
  imageUrl: z.string().optional(),
  selectedImageId: z.string().optional(),
});

type FormState = {
  message: string;
  predictions?: AllPredictionsResult;
  summary?: string;
  imagePreview?: string;
  error?: boolean;
};

// Helper to generate realistic mock probabilities for a given label
function getMockProbabilities(labels: string[], predictedLabel: string): PredictionResult['probabilities'] {
  let remainingProb = 1.0;
  
  // Assign high probability to the predicted label
  const predictedProb = Math.random() * 0.15 + 0.8; // 80% - 95%
  remainingProb -= predictedProb;

  const probabilities = [{ label: predictedLabel, prob: predictedProb, index: labels.indexOf(predictedLabel) }];
  
  // Get a few other random labels, ensuring they are not the predicted one
  const otherLabels = labels.filter(l => l !== predictedLabel);
  const shuffledLabels = otherLabels.sort(() => 0.5 - Math.random());

  // Distribute remaining probability among a few other labels
  for (let i = 0; i < 2; i++) {
      if (shuffledLabels[i]) {
          const prob = Math.random() * remainingProb * 0.6; // Distribute a portion of remaining
          probabilities.push({ label: shuffledLabels[i], prob, index: labels.indexOf(shuffledLabels[i]) });
          remainingProb -= prob;
      }
  }
  
  // Return top 3 probabilities sorted
  return probabilities.sort((a, b) => b.prob - a.prob).slice(0, 3);
}


export async function predictAllModels(
  prevState: FormState,
  formData: FormData
): Promise<FormState> {
  const validatedFields = schema.safeParse({
    image: formData.get('image'),
    imageUrl: formData.get('imageUrl'),
    selectedImageId: formData.get('selectedImageId')
  });

  if (!validatedFields.success) {
    return { message: 'Invalid input.', error: true };
  }
  
  const { image, imageUrl, selectedImageId } = validatedFields.data;

  if (!image && !imageUrl) {
    return { message: 'Please upload or select an image.', error: true };
  }
  
  if (image && image.size === 0 && !imageUrl) {
      return { message: 'Please upload an image.', error: true };
  }

  let imagePreview: string | undefined;
  if (imageUrl) {
    imagePreview = imageUrl;
  } else if (image) {
    const arrayBuffer = await image.arrayBuffer();
    const imageBuffer = Buffer.from(arrayBuffer);
    imagePreview = `data:${image.type};base64,${imageBuffer.toString('base64')}`;
  } else {
    return { message: 'No image provided.', error: true };
  }


  try {
    const [models, labels, testImages] = await Promise.all([getModels(), getLabels(), getTestImages()]);
    
    // --- SMART MOCK INFERENCE ---
    let predictedLabel: string;
    
    if (selectedImageId) {
        // If a sample image was used, "predict" its actual label
        const selectedTestImage = testImages.find(img => img.id === selectedImageId);
        predictedLabel = selectedTestImage ? selectedTestImage.description : "Strawberry Wedge 1";
    } else {
        // If an image was uploaded, default to a visually similar class from the list for a better demo
        predictedLabel = "Tomato Cherry Maroon 1";
    }

    if (!labels.includes(predictedLabel)) {
        // Fallback if the determined label doesn't exist in our labels list
        predictedLabel = labels[Math.floor(Math.random() * labels.length)];
    }
    
    // MOCK INFERENCE FOR ALL MODELS
    const predictionPromises = models.map(async (model): Promise<PredictionResult> => {
      const startTime = performance.now();
      
      const probabilities = getMockProbabilities(labels, predictedLabel);
      const endTime = performance.now();
      const inferenceTime = endTime - startTime + (Math.random() * 15 + 5);

      return {
        model_id: model.id,
        predicted_label: predictedLabel,
        predicted_index: labels.indexOf(predictedLabel),
        probabilities,
        inference_time_ms: parseFloat(inferenceTime.toFixed(1)),
      };
    });

    const predictionResults = await Promise.all(predictionPromises);
    const allPredictions: AllPredictionsResult = { results: predictionResults };
    
    // Call GenAI flow for summary
    const aiSummary = await summarizePredictionResults({ results: predictionResults });

    return {
      message: 'Prediction successful.',
      predictions: allPredictions,
      summary: aiSummary.summary,
      imagePreview,
    };
  } catch (e) {
    console.error(e);
    const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred during prediction.';
    return { message: errorMessage, error: true };
  }
}
