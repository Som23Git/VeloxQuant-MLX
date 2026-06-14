import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'VeloxQuant-MLX',
  tagline: '16× KV cache compression for Apple Silicon',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://veloxquant-mlx.netlify.app',
  baseUrl: '/docs/',

  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/rajveer43/turboquant_mac_implementation/edit/master/docs-site/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: true,
      respectPrefersColorScheme: false,
    },
    image: 'img/favicon.ico',
    navbar: {
      title: 'VeloxQuant-MLX',
      logo: {
        alt: 'VeloxQuant-MLX Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'gettingStartedSidebar',
          position: 'left',
          label: 'Getting Started',
        },
        {
          type: 'docSidebar',
          sidebarId: 'algorithmsSidebar',
          position: 'left',
          label: 'Algorithms',
        },
        {
          type: 'docSidebar',
          sidebarId: 'guidesSidebar',
          position: 'left',
          label: 'Guides',
        },
        {
          type: 'docSidebar',
          sidebarId: 'apiSidebar',
          position: 'left',
          label: 'API Reference',
        },
        {
          to: '/changelog',
          label: 'Changelog',
          position: 'left',
        },
        {
          href: 'https://github.com/rajveer43/turboquant_mac_implementation',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            { label: 'Getting Started', to: '/getting-started/intro' },
            { label: 'Algorithms', to: '/algorithms/overview' },
            { label: 'Guides', to: '/guides/mlx-lm-integration' },
            { label: 'API Reference', to: '/api/cache' },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/rajveer43/turboquant_mac_implementation',
            },
            {
              label: 'Issues',
              href: 'https://github.com/rajveer43/turboquant_mac_implementation/issues',
            },
          ],
        },
        {
          title: 'More',
          items: [
            { label: 'Changelog', to: '/changelog' },
            { label: 'Home', href: '/' },
          ],
        },
      ],
      copyright: `MIT License © ${new Date().getFullYear()} VeloxQuant-MLX. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.dracula,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'toml'],
    },
    algolia: {
      appId: process.env.YOUR_APP_ID!,
      apiKey: process.env.YOUR_SEARCH_API_KEY!,
      indexName: 'veloxquant-mlx',
      contextualSearch: true,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
