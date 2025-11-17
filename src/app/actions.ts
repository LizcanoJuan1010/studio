'use server';

// ================================
// Tipos de datos (coinciden con el backend)
// ================================
export type ProbabilityItem = {
  label: string;
  index: number;
  prob: number;
};

export type ModelPrediction = {
  model_id: string;
  predicted_label: string;
  predicted_index: number;
  probabilities: ProbabilityItem[];
  inference_time_ms: number;
};

export type PredictionResponse = ModelPrediction[];

// ================================
// Estado que usa LiveTestClient
// ================================
export type PredictionState = {
  message: string;
  error: boolean;
  predictions?: { results: PredictionResponse } | undefined;
  summary?: string | null;         // lo dejamos pero siempre será null
  imagePreview?: string | null;
};

// ================================
// Config de la API (FastAPI en Docker)
// ================================
const API_HOST = process.env.API_HOST ?? 'api';   // nombre del servicio en docker-compose
const API_PORT = process.env.API_PORT ?? '8000';
const API_BASE_URL = `http://${API_HOST}:${API_PORT}`;

// ================================
// Llamada genérica al backend
// ================================
async function callPredictAPI(fd: FormData): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE_URL}/predict/`, {
    method: 'POST',
    body: fd,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`API error ${res.status}: ${text || res.statusText}`);
  }

  return (await res.json()) as PredictionResponse;
}

// ================================
// Acción principal que usa LiveTestClient
// ================================
export async function predictAllModels(
  _prevState: PredictionState,
  formData: FormData
): Promise<PredictionState> {
  try {
    const fileEntry = formData.get('image') as any;
    const sampleDataUrl = formData.get('sampleDataUrl');

    let blob: Blob | null = null;
    let preview: string | null = null;

    // 1) Caso: archivo subido por el usuario
    if (
      fileEntry &&
      typeof fileEntry.arrayBuffer === 'function' &&
      typeof fileEntry.size === 'number' &&
      fileEntry.size > 0
    ) {
      const file = fileEntry as any;
      blob = file as Blob;

      const buffer = Buffer.from(await file.arrayBuffer());
      const mime = (file.type as string) || 'image/jpeg';
      const base64 = buffer.toString('base64');
      preview = `data:${mime};base64,${base64}`;
    }

    // 2) Caso: imagen de ejemplo → nos llega como dataURL desde el cliente
    else if (typeof sampleDataUrl === 'string' && sampleDataUrl.startsWith('data:')) {
      const [meta, base64data] = sampleDataUrl.split(',');
      const mimeMatch = /data:(.*);base64/.exec(meta);
      const mime = mimeMatch?.[1] || 'image/jpeg';

      const buffer = Buffer.from(base64data, 'base64');
      blob = new Blob([buffer], { type: mime });

      // Para la UI usamos el mismo dataURL
      preview = sampleDataUrl;
    }

    // 3) Ni archivo ni muestra
    else {
      return {
        message: 'No image provided (neither upload nor sample).',
        error: true,
        predictions: undefined,
        summary: null,
        imagePreview: null,
      };
    }

    // En este punto siempre tenemos un Blob válido
    const fd = new FormData();
    fd.append('file', blob!, 'image.jpg');

    // Llamar a FastAPI
    const results = await callPredictAPI(fd);

    return {
      message: 'Prediction successful.',
      error: false,
      predictions: { results },
      summary: null,          // 👈 nada de IA externa, siempre null
      imagePreview: preview,
    };
  } catch (e: any) {
    console.error('Error in predictAllModels:', e);
    return {
      message: e?.message ?? 'Unexpected error while predicting.',
      error: true,
      predictions: undefined,
      summary: null,
      imagePreview: null,
    };
  }
}
