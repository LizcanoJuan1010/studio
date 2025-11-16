'use server';

import { z } from 'zod';
import { getLabels, getModels } from '@/lib/data';
import type { AllPredictionsResult, PredictionResult, Model } from '@/lib/types';
import { summarizePredictionResults } from '@/ai/flows/summarize-prediction-results';

const schema = z.object({
  image: z.instanceof(File).optional(),
  imageUrl: z.string().optional(),
});

type FormState = {
  message: string;
  predictions?: AllPredictionsResult;
  summary?: string;
  imagePreview?: string;
  error?: boolean;
};

// Helper to shuffle array and get top probabilities
function getMockProbabilities(labels: string[], predictedLabel: string): PredictionResult['probabilities'] {
  let remainingProb = 1.0;
  
  // Assign high probability to the predicted label
  const predictedProb = Math.random() * 0.15 + 0.8; // 0.80 - 0.95
  remainingProb -= predictedProb;

  const probabilities = [{ label: predictedLabel, prob: predictedProb, index: labels.indexOf(predictedLabel) }];
  
  // Get a few other random labels, but ensure they are not the predicted label
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
  // Add a placeholder for the rest of the probability if needed, or just let it be.
  
  return probabilities.sort((a, b) => b.prob - a.prob);
}


export async function predictAllModels(
  prevState: FormState,
  formData: FormData
): Promise<FormState> {
  const validatedFields = schema.safeParse({
    image: formData.get('image'),
    imageUrl: formData.get('imageUrl'),
  });

  if (!validatedFields.success) {
    return { message: 'Invalid input.', error: true };
  }
  
  const { image, imageUrl } = validatedFields.data;

  if (!image && !imageUrl) {
    return { message: 'Please upload or select an image.', error: true };
  }
  
  if (image && image.size === 0 && !imageUrl) {
      return { message: 'Please upload an image.', error: true };
  }

  let imageBuffer: Buffer;
  let imagePreview: string | undefined;

  if (imageUrl) {
    // Handle sample images (Data URL)
    const base64Data = imageUrl.split(',')[1];
    imageBuffer = Buffer.from(base64Data, 'base64');
    imagePreview = imageUrl;
  } else if (image) {
    // Handle uploaded file
    const arrayBuffer = await image.arrayBuffer();
    imageBuffer = Buffer.from(arrayBuffer);
    imagePreview = `data:${image.type};base64,${imageBuffer.toString('base64')}`;
  } else {
    return { message: 'No image provided.', error: true };
  }


  try {
    const [models, labels] = await Promise.all([getModels(), getLabels()]);
    
    // MOCK INFERENCE FOR ALL MODELS - Forcing a more realistic prediction
    const predictionPromises = models.map(async (model): Promise<PredictionResult> => {
      const startTime = performance.now();
      
      // Force prediction to be Strawberry for this demonstration
      const predictedLabel = "Strawberry Wedge 1";
      if (!labels.includes(predictedLabel)) {
        // Fallback if the label doesn't exist
        const randomLabel = labels[Math.floor(Math.random() * labels.length)];
        const probabilities = getMockProbabilities(labels, randomLabel);
        const endTime = performance.now();
        const inferenceTime = endTime - startTime + (Math.random() * 15 + 5);
        return {
          model_id: model.id,
          predicted_label: randomLabel,
          predicted_index: labels.indexOf(randomLabel),
          probabilities,
          inference_time_ms: parseFloat(inferenceTime.toFixed(1)),
        };
      }
      
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
