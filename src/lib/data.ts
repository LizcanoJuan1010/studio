import 'server-only';
import fs from 'fs/promises';
import path from 'path';
import { AllGlobalMetrics, ConfusionMatrix, Model, PerClassMetrics } from './types';

const dataPath = path.join(process.cwd(), 'src', 'data');
// Test directory path - may not exist in Docker builds (excluded by .dockerignore)
// This is fine, the function will return an empty array if the directory doesn't exist
const testImagePath = path.join(process.cwd(), 'src', 'Test');


async function readJsonFile<T>(filename: string): Promise<T> {
  try {
    const filePath = path.join(dataPath, filename);
    const fileContent = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(fileContent);
  } catch (error) {
    console.error(`Error reading or parsing ${filename}:`, error);
    // If a file for a specific model doesn't exist, we can return a default/empty state
    // For now, we'll re-throw to make it clear a file is missing.
    throw new Error(`Could not load data for ${filename}.`);
  }
}

export async function getModels(): Promise<Model[]> {
  return readJsonFile<Model[]>('models.json');
}

export async function getModelById(id: string): Promise<Model | undefined> {
    const models = await getModels();
    return models.find(m => m.id === id);
}

export async function getGlobalMetrics(): Promise<AllGlobalMetrics> {
  return readJsonFile<AllGlobalMetrics>('metrics-global.json');
}

export async function getPerClassMetrics(modelId: string): Promise<PerClassMetrics | null> {
  const fileName = `metrics-${modelId}-per-class.json`;
  try {
    return await readJsonFile<PerClassMetrics>(fileName);
  } catch (error) {
    // It's okay if the file doesn't exist, we'll handle it in the component.
    return null;
  }
}

export async function getConfusionMatrix(modelId: string): Promise<ConfusionMatrix> {
    const fileName = `confusion-matrix-${modelId}.json`;
    try {
        return await readJsonFile<ConfusionMatrix>(fileName);
    } catch (error) {
        console.warn(`Warning: Could not find ${fileName}. Returning mock data.`);
        const labels = (await getLabels()).slice(0, 10);
        return {
            labels,
            matrix: labels.map(() => labels.map(() => Math.floor(Math.random() * 5)))
        }
    }
}

export async function getLabels(): Promise<string[]> {
    return readJsonFile<string[]>('labels.json');
}

export async function getTestImages(): Promise<{ id: string, description: string, imageUrl: string }[]> {
  try {
    // Limit the number of directories processed to avoid memory issues
    const maxDirs = 50; // Process only first 50 fruit directories
    const fruitDirs = (await fs.readdir(testImagePath)).slice(0, maxDirs);
    
    const imagePromises = fruitDirs.map(async (fruitDir) => {
      try {
        const dirPath = path.join(testImagePath, fruitDir);
        const stats = await fs.stat(dirPath);
        if (!stats.isDirectory()) return null;
        
        const files = await fs.readdir(dirPath);
        const imageFile = files.find(file => /\.(jpg|jpeg|png)$/i.test(file));
        
        if (imageFile) {
          const imagePath = path.join(dirPath, imageFile);
          const imageBuffer = await fs.readFile(imagePath);
          // Limit image size to prevent memory issues (max 2MB)
          if (imageBuffer.length > 2 * 1024 * 1024) {
            console.warn(`Skipping large image: ${fruitDir}/${imageFile} (${Math.round(imageBuffer.length / 1024)}KB)`);
            return null;
          }
          const imageBase64 = imageBuffer.toString('base64');
          const mimeType = path.extname(imageFile).slice(1).toLowerCase() === 'jpg' ? 'jpeg' : path.extname(imageFile).slice(1).toLowerCase();
          return {
            id: fruitDir.replace(/\s/g, '-').toLowerCase(),
            description: fruitDir,
            imageUrl: `data:image/${mimeType};base64,${imageBase64}`,
          };
        }
        return null;
      } catch (dirError) {
        // Skip directories that can't be read
        console.warn(`Skipping directory ${fruitDir}:`, dirError);
        return null;
      }
    });

    const images = (await Promise.all(imagePromises)).filter(Boolean);
    // @ts-ignore
    return images;
  } catch (error) {
    console.error('Failed to read test images:', error);
    // Return empty array instead of failing
    return [];
  }
}

    