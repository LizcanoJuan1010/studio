'use server';

import { z } from 'zod';
import { getLabels, getModels } from '@/lib/data';
import type { AllPredictionsResult, PredictionResult } from '@/lib/types';
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
  const predictedProb = Math.random() * 0.2 + 0.75; // 0.75 - 0.95
  remainingProb -= predictedProb;

  const probabilities = [{ label: predictedLabel, prob: predictedProb, index: labels.indexOf(predictedLabel) }];
  
  const otherLabels = labels.filter(l => l !== predictedLabel);

  // Distribute remaining probability among other labels
  for (let i = 0; i < otherLabels.length - 1; i++) {
    const prob = Math.random() * remainingProb * 0.5;
    probabilities.push({ label: otherLabels[i], prob, index: labels.indexOf(otherLabels[i]) });
    remainingProb -= prob;
  }
  probabilities.push({ label: otherLabels[otherLabels.length - 1], prob: remainingProb, index: labels.indexOf(otherLabels[otherLabels.length - 1]) });

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
    const base64Data = imageUrl.split(',')[1];
    imageBuffer = Buffer.from(base64Data, 'base64');
    imagePreview = imageUrl;
  } else if (image) {
    const arrayBuffer = await image.arrayBuffer();
    imageBuffer = Buffer.from(arrayBuffer);
    imagePreview = `data:${image.type};base64,${imageBuffer.toString('base64')}`;
  } else {
    return { message: 'No image provided.', error: true };
  }


  try {
    const [models, labels] = await Promise.all([getModels(), getLabels()]);
    
    const predictionPromises = models.map(async (model): Promise<PredictionResult> => {
      const startTime = performance.now();
      
      // MOCK INFERENCE FOR ALL MODELS
      const predictedLabel = labels[Math.floor(Math.random() * labels.length)];
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
    
    // Call GenAI flow
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
