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
function getProbabilities(labels: string[], predictedLabel: string): PredictionResult['probabilities'] {
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


  try {
    const [models, labels] = await Promise.all([getModels(), getLabels()]);

    const predictionResults: PredictionResult[] = models.map((model) => {
      const startTime = performance.now();
      
      // --- MOCK INFERENCE ---
      const predictedLabel = labels[Math.floor(Math.random() * labels.length)];
      const probabilities = getProbabilities(labels, predictedLabel);
      // --- END MOCK INFERENCE ---

      const endTime = performance.now();
      const inferenceTime = endTime - startTime + (Math.random() * 10); // Add random base time

      return {
        model_id: model.id,
        predicted_label: predictedLabel,
        predicted_index: labels.indexOf(predictedLabel),
        probabilities,
        inference_time_ms: parseFloat(inferenceTime.toFixed(1)),
      };
    });

    const allPredictions: AllPredictionsResult = { results: predictionResults };
    
    // Call GenAI flow
    const aiSummary = await summarizePredictionResults({ results: predictionResults });

    let imagePreview: string | undefined;

    if (imageUrl) {
      imagePreview = imageUrl;
    } else if (image) {
      const arrayBuffer = await image.arrayBuffer();
      imagePreview = `data:${image.type};base64,${Buffer.from(arrayBuffer).toString('base64')}`;
    }

    return {
      message: 'Prediction successful.',
      predictions: allPredictions,
      summary: aiSummary.summary,
      imagePreview,
    };
  } catch (e) {
    console.error(e);
    return { message: 'An error occurred during prediction.', error: true };
  }
}
