'use server';

import { z } from 'zod';
import type { AllPredictionsResult, PredictionResult } from '@/lib/types';
import { summarizePredictionResults } from '@/ai/flows/summarize-prediction-results';

// Use a custom File validation that works in server-side Next.js
const fileSchema = z.any().optional().refine(
  (val) => {
    if (!val) return true; // Optional, so undefined/null is OK
    // In Next.js Server Actions, FormData File objects have type and size
    return typeof val === 'object' && 
           'name' in val && 
           'size' in val && 
           'type' in val &&
           val instanceof Object;
  },
  { message: "Invalid file object" }
);

const schema = z.object({
  image: fileSchema,
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
  
  const { image, imageUrl } = validatedFields.data;

  if ((!image || image.size === 0) && !imageUrl) {
    return { message: 'Please upload or select an image.', error: true };
  }

  let fileToUpload: File;

  if (imageUrl) {
    try {
      // For sample images, fetch the data from the base64 URL
      const response = await fetch(imageUrl);
      if (!response.ok) throw new Error('Failed to fetch sample image');
      const blob = await response.blob();
      fileToUpload = new File([blob], 'image.jpg', { type: blob.type });
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'Unknown error fetching sample image.';
      return { message: `Could not load sample image: ${errorMessage}`, error: true };
    }
  } else if (image) {
    fileToUpload = image;
  } else {
    return { message: 'No image provided.', error: true };
  }
  
  const apiFormData = new FormData();
  apiFormData.append('file', fileToUpload);

  try {
    // Docker networking: 'api' is the service name, port 8000 is internal
    const apiHost = process.env.API_HOST || 'localhost';
    const apiPort = process.env.API_PORT || '8001';
    const apiUrl = `http://${apiHost}:${apiPort}/predict/`;
    
    console.log(`Calling prediction API at: ${apiUrl}`);
    
    const response = await fetch(apiUrl, {
        method: 'POST',
        body: apiFormData,
    });

    if (!response.ok) {
        const errorBody = await response.text();
        console.error('API Error Response:', errorBody);
        throw new Error(`Prediction request failed with status ${response.status}: ${response.statusText}`);
    }
    
    const predictionResults: PredictionResult[] = await response.json();
    
    const allPredictions: AllPredictionsResult = { results: predictionResults };
    
    // Call GenAI flow for summary
    const aiSummary = await summarizePredictionResults({ results: predictionResults });
    
    // Convert local URL to base64 for state to avoid client-side issues
    const imageBuffer = await fileToUpload.arrayBuffer();
    const imageBase64 = Buffer.from(imageBuffer).toString('base64');
    const imagePreviewForState = `data:${fileToUpload.type};base64,${imageBase64}`;

    return {
      message: 'Prediction successful.',
      predictions: allPredictions,
      summary: aiSummary.summary,
      imagePreview: imagePreviewForState,
    };
  } catch (e) {
    console.error('Error during prediction fetch:', e);
    const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred.';
    return { message: `Failed to connect to the prediction backend. Make sure the backend service is running. Details: ${errorMessage}`, error: true };
  }
}
