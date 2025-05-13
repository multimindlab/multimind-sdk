// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'MultiMind RAG System',
  tagline: 'Advanced Retrieval-Augmented Generation System',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://multimind.dev',
  // Set the /<baseUrl>/ pathname under which your site is served
  baseUrl: '/',

  // GitHub pages deployment config
  organizationName: 'multimind-dev',
  projectName: 'multimind-sdk',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/multimind-dev/multimind-sdk/edit/main/website/',
          showLastUpdateTime: true,
        },
        blog: {
          showReadingTime: true,
          editUrl: 'https://github.com/multimind-dev/multimind-sdk/edit/main/website/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/multimind-social-card.jpg',
      navbar: {
        title: 'MultiMind RAG',
        logo: {
          alt: 'MultiMind Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Documentation',
          },
          {
            to: '/features',
            label: 'Features',
            position: 'left',
          },
          {
            to: '/api',
            label: 'API Reference',
            position: 'left',
          },
          {
            to: '/architecture',
            label: 'Architecture',
            position: 'left',
          },
          {
            href: 'https://github.com/multimind-dev/multimind-sdk',
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
              {
                label: 'Getting Started',
                to: '/docs/getting-started',
              },
              {
                label: 'API Reference',
                to: '/api',
              },
              {
                label: 'Architecture',
                to: '/architecture',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Discord',
                href: 'https://discord.gg/multimind',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/multimind_dev',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/multimind-dev/multimind-sdk',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/multimind-dev/multimind-sdk',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} MultiMind. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        additionalLanguages: ['python', 'bash', 'json'],
      },
      // Enable Algolia DocSearch
      algolia: {
        appId: 'YOUR_APP_ID',
        apiKey: 'YOUR_API_KEY',
        indexName: 'multimind',
        contextualSearch: true,
      },
    }),
};

module.exports = config; 