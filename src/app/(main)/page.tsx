import { ModelCard } from '@/components/model-card';
import { getGlobalMetrics, getModels } from '@/lib/data';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"

export default async function HomePage() {
  const models = await getModels();
  const allMetrics = await getGlobalMetrics();

  return (
    <div className="flex flex-col gap-8">
      <div className="animate-fade-in-down">
        <h1 className="text-4xl font-bold tracking-tight font-headline animate-text-gradient bg-gradient-to-r from-primary via-accent to-primary bg-[200%_auto]">
          Fruit Classification Model Comparator
        </h1>
        <p className="text-muted-foreground mt-2">
          An interactive dashboard to visualize and compare the performance of different AI models for fruit classification.
        </p>
      </div>

      <Card className="animate-fade-in" style={{ animationDelay: '200ms' }}>
          <CardHeader>
              <CardTitle className="font-headline text-xl">Descripción del Proyecto</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-muted-foreground">
              <p>
                  El desarrollo de modelos de aprendizaje automático ha permitido abordar problemas de predicción y clasificación en múltiples dominios, uno de los cuales es la visión artificial que busca trabajar para extraer información y enseñarle a la maquina a partir de imágenes y videos. En este proyecto se desarrolla un sistema de clasificación de imágenes para frutas, verduras, semillas y nueces que cuenta en total con 221 clases. Se plantea comparar diferentes enfoques y modelos a lo largo del proyecto: CNN desde cero, CNN con transfer learning, SVM y Boosting. La selección del modelo se basará en accuracy y F1 macro, complementada con matrices de confusión. Al final del proyecto se entregará un demo de inferencia.
              </p>
              <Accordion type="multiple" className="w-full">
                <AccordionItem value="item-1">
                  <AccordionTrigger className="font-semibold text-foreground text-base">Objetivo general:</AccordionTrigger>
                  <AccordionContent>
                    Construir un modelo de clasificación de frutas a partir del dataset Fruits-360, aplicando enfoques de aprendizaje de máquina y criterios básicos de evaluación.
                  </AccordionContent>
                </AccordionItem>
                <AccordionItem value="item-2">
                  <AccordionTrigger className="font-semibold text-foreground text-base">Objetivos específicos:</AccordionTrigger>
                  <AccordionContent>
                    <ul className="list-disc pl-5 space-y-2">
                        <li>Explorar y caracterizar los datos para identificar variables relevantes. Desarrollar una comprensión global del dataset Fruits-360 mediante análisis descriptivo, distribución de clases y detección de sesgos.</li>
                        <li>Definir y entrenar modelos de clasificación adecuados al contexto del problema. Seleccionar enfoques de modelado acordes con la naturaleza de los datos, priorizando estrategias simples y transparentes.</li>
                        <li>Evaluar el desempeño con métricas estándar de clasificación. Valorar la calidad de las predicciones usando indicadores reconocidos (exactitud, precisión, recall, F1) bajo un esquema de validación.</li>
                        <li>Implementar una aplicación sencilla para visualizar los resultados del clasificador. Diseñar una app básica donde se puedan ingresar ejemplos tomados del mismo dataset y observar el resultado de la clasificación de manera interactiva.</li>
                    </ul>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
          </CardContent>
      </Card>
      
      <Card className="animate-fade-in" style={{ animationDelay: '400ms' }}>
          <CardHeader>
              <CardTitle className="font-headline text-xl">Descripción del Dataset</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-muted-foreground">
              <p>
                  El proyecto se desarrollará con el dataset “Fruits-360” (Otlean, 2018), el cual fue extraído de la plataforma Kaggle, y es un conjunto de imágenes de frutas, verduras, nueces y semillas que incluye múltiples variedades por cada clase (como" Apple Red Delicious” o “Apple Granny Smith”), totas las imágenes fueron tomadas en un fondo blanco y se segmentaron del fondo para facilitar el entrenamiento de modelos. El dataset cuentas con un total de 152 665 imágenes (114 482 train / 38 183 test), 221 clases y tamaño 100×100 pixeles.
              </p>
          </CardContent>
      </Card>

      <div>
        <h2 className="text-2xl font-bold tracking-tight font-headline animate-fade-in" style={{ animationDelay: '600ms' }}>
          Modelos Entrenados
        </h2>
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 mt-4">
          {models.map((model, index) => (
            <ModelCard 
              key={model.id} 
              model={model} 
              metrics={allMetrics[model.id]} 
              style={{ animationDelay: `${index * 150 + 700}ms` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}