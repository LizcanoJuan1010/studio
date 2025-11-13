import { ModelDetailClient } from '@/components/model-detail-client';
import { getModelById, getGlobalMetrics, getPerClassMetrics, getConfusionMatrix, getModels } from '@/lib/data';
import { notFound } from 'next/navigation';

type ModelPageProps = {
  params: {
    model_id: string;
  };
};

export async function generateMetadata({ params }: ModelPageProps) {
    try {
        const model = await getModelById(params.model_id);
        if (!model) {
            return { title: 'Model Not Found' };
        }
        return { title: `${model.name} Details` };
    } catch (error) {
        return { title: 'Model Not Found' };
    }
}

export default async function ModelPage({ params }: ModelPageProps) {
  const { model_id } = params;

  const model = await getModelById(model_id);
  if (!model) {
    notFound();
  }

  try {
    const [allGlobalMetrics, perClassMetrics, confusionMatrix] = await Promise.all([
      getGlobalMetrics(),
      getPerClassMetrics(model_id),
      getConfusionMatrix(model_id),
    ]);
    
    const globalMetrics = allGlobalMetrics[model.id as keyof typeof allGlobalMetrics];

    if (!globalMetrics || !perClassMetrics || !confusionMatrix) {
        // This case handles if a data file is missing for a valid model id
        notFound();
    }

    return (
        <ModelDetailClient 
            model={model}
            globalMetrics={globalMetrics}
            perClassMetrics={perClassMetrics}
            confusionMatrix={confusionMatrix}
        />
    );
  } catch (error) {
    console.error(`Failed to load data for model ${model_id}:`, error);
    notFound();
  }
}

export async function generateStaticParams() {
    try {
        const models = await getModels();
        return models.map((model) => ({
          model_id: model.id,
        }));
    } catch (error) {
        console.error('Failed to generate static params for models:', error);
        return [];
    }
}
