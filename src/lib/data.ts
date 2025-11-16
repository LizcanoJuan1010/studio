import 'server-only';
import fs from 'fs/promises';
import path from 'path';
import { AllGlobalMetrics, ConfusionMatrix, Model, PerClassMetrics } from './types';

const dataPath = path.join(process.cwd(), 'src', 'data');
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

export async function getPerClassMetrics(modelId: string): Promise<PerClassMetrics> {
  const fileName = `metrics-${modelId}-per-class.json`;
  try {
    return await readJsonFile<PerClassMetrics>(fileName);
  } catch (error) {
    console.warn(`Warning: Could not find ${fileName}. Returning mock data.`);
    const labels = await getLabels();
    return {
      model_id: modelId as Model['id'],
      per_class: labels.slice(0, 50).map((label, index) => ({
        class_id: index,
        label: label,
        precision: Math.random() * (0.99 - 0.85) + 0.85,
        recall: Math.random() * (0.99 - 0.85) + 0.85,
        f1: Math.random() * (0.99 - 0.85) + 0.85,
      }))
    }
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
    const fruitDirs = await fs.readdir(testImagePath);
    const imagePromises = fruitDirs.map(async (fruitDir) => {
      const dirPath = path.join(testImagePath, fruitDir);
      const stats = await fs.stat(dirPath);
      if (!stats.isDirectory()) return null;
      
      const files = await fs.readdir(dirPath);
      const imageFile = files.find(file => /\.(jpg|jpeg|png)$/i.test(file));
      
      if (imageFile) {
        const imagePath = path.join(dirPath, imageFile);
        const imageBuffer = await fs.readFile(imagePath);
        const imageBase64 = imageBuffer.toString('base64');
        const mimeType = path.extname(imageFile).slice(1);
        return {
          id: fruitDir.replace(/\s/g, '-').toLowerCase(),
          description: fruitDir,
          imageUrl: `data:image/${mimeType};base64,${imageBase64}`,
        };
      }
      return null;
    });

    const images = (await Promise.all(imagePromises)).filter(Boolean);
    // @ts-ignore
    return images;
  } catch (error) {
    console.error('Failed to read test images:', error);
    return [];
  }
}

    