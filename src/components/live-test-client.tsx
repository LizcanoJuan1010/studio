
'use client';

import { useActionState, useRef, useEffect, useState } from 'react';
import { predictAllModels } from '@/app/actions';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { UploadCloud, LoaderCircle, Sparkles, XCircle } from 'lucide-react';
import Image from 'next/image';
import { PredictionCard } from './prediction-card';
import { useToast } from '@/hooks/use-toast';
import { TestImage } from '@/lib/types';
import { cn } from '@/lib/utils';
import { Separator } from './ui/separator';

const initialState = {
  message: '',
  error: false,
};

export function LiveTestClient({ testImages }: { testImages: TestImage[] }) {
  const [state, formAction, isPending] = useActionState(predictAllModels, initialState);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const { toast } = useToast();
  const [selectedImage, setSelectedImage] = useState<TestImage | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  useEffect(() => {
    if (state.message && state.error) {
        toast({
            variant: "destructive",
            title: "Prediction Error",
            description: state.message,
        })
    }
  }, [state, toast]);

  const handleImageSelect = (image: TestImage) => {
    setSelectedImage(image);
    setImagePreview(image.imageUrl);
    // Clear the file input if a sample image is selected
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFormSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    
    const file = formData.get('image') as File;
    const hasFile = file && file.size > 0;

    if (!selectedImage && !hasFile) {
        toast({
            variant: "destructive",
            title: "No Image Selected",
            description: "Please select a sample image or upload your own.",
        });
        return;
    }

    if (selectedImage) {
      formData.set('imageUrl', selectedImage.imageUrl);
      if (hasFile) {
        formData.delete('image');
      }
    }
    formAction(formData);
  };
  
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(null);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };


  return (
    <div className="flex flex-col gap-8 animate-fade-in-up">
      <div>
        <h1 className="text-3xl font-bold tracking-tight font-headline">
          Live Prediction Test
        </h1>
        <p className="text-muted-foreground mt-2">
          Upload an image of a fruit or select a sample image to get predictions from all available models in real-time.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="font-headline text-xl">Select a Test Image</CardTitle>
          <CardDescription>Choose one of the sample images below to test the models.</CardDescription>
        </CardHeader>
        <CardContent>
            {testImages.length > 0 ? (
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-4">
                    {testImages.map((image) => (
                        <button 
                            key={image.id}
                            onClick={() => handleImageSelect(image)}
                            className={cn(
                                "relative aspect-square w-full rounded-lg overflow-hidden border-2 transition-all",
                                selectedImage?.id === image.id ? 'border-primary ring-2 ring-primary ring-offset-2' : 'border-border hover:border-primary'
                            )}
                        >
                            <Image 
                                src={image.imageUrl} 
                                alt={image.description} 
                                fill 
                                className="object-cover"
                                sizes="(max-width: 768px) 50vw, (max-width: 1024px) 25vw, 12.5vw"
                            />
                        </button>
                    ))}
                </div>
            ) : (
                <div className="flex flex-col items-center justify-center rounded-lg border border-dashed p-12 text-center">
                    <XCircle className="mx-auto h-12 w-12 text-muted-foreground" />
                    <h3 className="mt-4 text-lg font-semibold">No Test Images Found</h3>
                    <p className="mt-1 text-sm text-muted-foreground">
                        Could not load images from the `src/lib/Test` directory.
                    </p>
                </div>
            )}
        </CardContent>
      </Card>
      
      <div className="relative flex items-center justify-center">
        <Separator className="w-full" />
        <span className="absolute bg-background px-4 text-sm text-muted-foreground font-medium">OR</span>
      </div>


      <Card>
        <CardHeader>
          <CardTitle className="font-headline text-xl">Upload Your Own Image</CardTitle>
        </CardHeader>
        <CardContent>
          <form ref={formRef} onSubmit={handleFormSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="image">Fruit Image</Label>
              <Input id="image" name="image" type="file" accept="image/*" ref={fileInputRef} onChange={handleFileChange} />
            </div>
            
            {imagePreview && !isPending && (
                <div className="relative aspect-square max-w-xs mx-auto w-full animate-fade-in">
                    <Image 
                        src={imagePreview} 
                        alt="Image preview" 
                        fill 
                        className="rounded-lg object-contain"
                    />
                </div>
            )}

            <Button type="submit" disabled={isPending} className="w-full md:w-auto">
              {isPending ? (
                <>
                  <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
                  Predicting...
                </>
              ) : (
                'Run Prediction'
              )}
            </Button>
          </form>
        </CardContent>
      </Card>
      
      {isPending && (
          <div className="flex items-center justify-center rounded-lg border border-dashed p-12 text-center animate-fade-in">
              <div className="flex flex-col items-center gap-2 text-muted-foreground">
                <LoaderCircle className="h-10 w-10 animate-spin text-primary"/>
                <p className="text-lg font-medium">Analyzing image...</p>
                <p className="text-sm">Models are warming up. This might take a moment.</p>
              </div>
          </div>
      )}

      {state.predictions && state.imagePreview && (
        <Card className="animate-fade-in">
            <CardHeader>
                <CardTitle className="font-headline text-xl">Prediction Results</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-8">
                <div className="grid gap-6 lg:grid-cols-[1fr_2fr]">
                    <div className="relative aspect-square max-w-sm mx-auto w-full">
                        <Image 
                            src={state.imagePreview} 
                            alt="Uploaded fruit" 
                            fill 
                            className="rounded-lg object-cover"
                        />
                    </div>
                    <div className="flex flex-col gap-4">
                        <h3 className="font-semibold font-headline">Model Outputs</h3>
                        <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-1 xl:grid-cols-2">
                            {state.predictions.results.map((result, index) => (
                                <PredictionCard 
                                    key={result.model_id} 
                                    result={result} 
                                    style={{ animationDelay: `${index * 100}ms` }}
                                />
                            ))}
                        </div>
                    </div>
                </div>

                {state.summary && (
                    <Alert className="bg-primary/5 border-primary/20 animate-fade-in" style={{ animationDelay: '300ms' }}>
                        <Sparkles className="h-4 w-4 text-primary" />
                        <AlertTitle className="font-headline text-primary">AI Summary</AlertTitle>
                        <AlertDescription className="text-primary/90">
                            {state.summary}
                        </AlertDescription>
                    </Alert>
                )}
            </CardContent>
        </Card>
      )}

      {!isPending && !state.predictions && (
        <div className="flex flex-col items-center justify-center rounded-lg border border-dashed p-12 text-center animate-fade-in">
          <UploadCloud className="mx-auto h-12 w-12 text-muted-foreground" />
          <h3 className="mt-4 text-lg font-semibold">Awaiting Image</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Your prediction results will appear here once you upload an image.
          </p>
        </div>
      )}
    </div>
  );
}
