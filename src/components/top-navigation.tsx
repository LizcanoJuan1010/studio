
'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Home,
  GitCompareArrows,
  BrainCircuit,
  Spline,
  Rocket,
  Beaker,
  Grape,
  Menu,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Button } from './ui/button';

const links = [
  { href: '/', label: 'Home', icon: Home },
  { href: '/compare-models', label: 'Compare Models', icon: GitCompareArrows },
  { href: '/models/cnn', label: 'CNN Model', icon: BrainCircuit },
  { href: '/models/svm', label: 'SVM Model', icon: Spline },
  { href: '/models/boosting', label: 'Boosting Model', icon: Rocket },
  { href: '/live-test', label: 'Live Test', icon: Beaker },
];

export function TopNavigation() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 flex h-16 items-center gap-4 border-b bg-background px-4 md:px-6 z-50">
      <nav className="hidden flex-col gap-6 text-lg font-medium md:flex md:flex-row md:items-center md:gap-5 md:text-sm lg:gap-6">
        <Link
          href="/"
          className="flex items-center gap-2 text-lg font-semibold md:text-base text-primary"
        >
          <Grape className="h-6 w-6" />
          <span className="sr-only">FruitVision</span>
        </Link>
        {links.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className={cn(
                "transition-colors hover:text-foreground",
                pathname === link.href || (link.href !== '/' && pathname.startsWith(link.href)) ? 'text-foreground font-semibold' : 'text-muted-foreground'
            )}
          >
            {link.label}
          </Link>
        ))}
      </nav>
      <div className="md:hidden">
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon">
                    <Menu className="h-5 w-5" />
                    <span className="sr-only">Toggle navigation menu</span>
                </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start">
                {links.map((link) => (
                    <DropdownMenuItem key={link.href} asChild>
                         <Link
                            href={link.href}
                            className="flex items-center gap-2"
                        >
                            <link.icon className="h-4 w-4" />
                            {link.label}
                        </Link>
                    </DropdownMenuItem>
                ))}
            </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <div className="flex w-full items-center gap-4 md:ml-auto md:flex-initial">
        {/* Future elements like search or user menu can go here */}
      </div>
    </header>
  );
}
