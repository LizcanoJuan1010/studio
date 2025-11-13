'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Home,
  GitCompareArrows,
  Search,
  User,
  Grape,
  Menu,
  BarChart,
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
  { href: '/compare-models', label: 'Compare', icon: GitCompareArrows },
  { href: '/live-test', label: 'Live Test', icon: Search },
];

export function TopNavigation() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 px-4 md:px-6">
      <div className="flex h-20 items-center justify-between rounded-full border border-white/10 bg-black/30 px-6 my-4 backdrop-blur-sm">
        <Link href="/" className="flex items-center gap-2 text-lg font-semibold text-primary-foreground md:text-base">
          <Grape className="h-6 w-6 text-primary" />
          <span className="font-bold text-white hidden md:inline">FruitVision</span>
        </Link>
        
        {/* Desktop Navigation */}
        <nav className="hidden md:flex h-12 items-center justify-center gap-2 rounded-full">
          {links.map((link) => {
            const isActive = link.href === '/' ? pathname === '/' : pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                    "flex items-center justify-center rounded-full px-4 py-2 text-sm font-medium transition-colors h-10 w-auto",
                    isActive
                    ? 'bg-primary text-primary-foreground shadow-lg'
                    : 'text-gray-300 hover:bg-white/10 hover:text-white'
                )}
                aria-current={isActive ? 'page' : undefined}
              >
                <link.icon className="h-5 w-5" />
                {isActive && <span className="ml-2">{link.label}</span>}
              </Link>
            )
          })}
        </nav>
        
        <div className="hidden md:flex">
             <Link
                href="#"
                className="flex items-center justify-center rounded-full px-4 py-2 text-sm font-medium transition-colors h-10 w-auto text-gray-300 hover:bg-white/10 hover:text-white"
              >
                <User className="h-5 w-5" />
              </Link>
        </div>

        {/* Mobile Navigation */}
        <div className="md:hidden">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="text-white hover:bg-white/10">
                <Menu className="h-5 w-5" />
                <span className="sr-only">Toggle navigation menu</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="border-white/10 bg-black/50 text-white backdrop-blur-lg">
              {[...links, {href: "#", label: "Profile", icon: User}].map((link) => (
                <DropdownMenuItem key={link.href} asChild>
                  <Link href={link.href} className="flex items-center gap-2">
                    <link.icon className="h-4 w-4" />
                    <span>{link.label}</span>
                  </Link>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
}
