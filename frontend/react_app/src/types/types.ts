import { LucideIcon } from 'lucide-react';

export interface NavLink {
  label: string;
  href: string;
}

export interface FeatureTab {
  id: string;
  label: string;
  icon: LucideIcon;
  color: string;
  title: string;
  description: string;
  benefits: string[];
  mockupData: MockupData;
}

export interface MockupData {
  type: 'mail' | 'chat' | 'doc' | 'task';
  items: Array<{
    id: number;
    title: string;
    subtitle?: string;
    tag?: string;
    avatar?: string;
    content?: string;
  }>;
}
