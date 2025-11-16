'use server';

import { z } from 'zod';
import { getLabels, getTestImages } from '@/lib/data';
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

  if ((!image || image.size === 0) && !imageUrl) {
    return { message: 'Please upload or select an image.', error: true };
  }

  let imagePreview: string | undefined;
  let apiFormData = new FormData();

  if (imageUrl) {
    imagePreview = imageUrl;
    // Since we're using a sample image, we need to fetch it and create a File object
    const response = await fetch(imageUrl);
    const blob = await response.blob();
    const fileName = selectedImageId || 'image.jpg';
    const file = new File([blob], fileName, { type: blob.type });
    apiFormData.append('file', file);

  } else if (image) {
    const arrayBuffer = await image.arrayBuffer();
    const imageBuffer = Buffer.from(arrayBuffer);
    imagePreview = `data:${image.type};base64,${imageBuffer.toString('base64')}`;
    apiFormData.append('file', image);
  } else {
    return { message: 'No image provided.', error: true };
  }


  try {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://api:8000';
    console.log(`Calling API at: ${apiUrl}/predict/`);

    const response = await fetch(`${apiUrl}/predict/`, {
        method: 'POST',
        body: apiFormData,
    });

    if (!response.ok) {
        const errorBody = await response.text();
        console.error('API Error Response:', errorBody);
        throw new Error(`Prediction failed: ${response.statusText}`);
    }
    
    const predictionResults: PredictionResult[] = await response.json();
    
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
    return { message: `Failed to connect to the prediction backend. ${errorMessage}`, error: true };
  }
}
