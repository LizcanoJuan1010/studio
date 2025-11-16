import { LiveTestClient } from "@/components/live-test-client";
import { getTestImages } from "@/lib/data";

export const metadata = {
    title: 'Live Test',
};

// Force dynamic rendering to ensure Test directory is available at runtime
export const dynamic = 'force-dynamic';

export default async function LiveTestPage() {
    const testImages = await getTestImages();
    console.log(`Loaded ${testImages.length} test images`);
    return <LiveTestClient testImages={testImages} />;
}
