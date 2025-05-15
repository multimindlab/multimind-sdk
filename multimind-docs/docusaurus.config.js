// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'MultiMind SDK',
  tagline: 'One SDK.Every AI Model.Unlimited Agents',

  // Set the production url of your site here
  url: 'https://multimind.dev',
  // Set the /<baseUrl>/ pathname under which your site is served
  baseUrl: '/',

  // GitHub pages deployment config
  organizationName: 'Ai2Innovate',
  projectName: 'multimind-sdk',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'fr', 'nl', 'pt'],
    localeConfigs: {
      en: { label: 'English' },
      fr: { label: 'Français' },
      nl: { label: 'Nederlands' },
      pt: { label: 'Português' },
    },
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/multimind-dev/multimind-sdk/edit/develop/',
          showLastUpdateTime: true,
        },
        blog: {
          showReadingTime: true,
          editUrl: 'https://www.multimind.dev/blogs',
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
      image: '../assets/Logo-with-name-final2.png',
      navbar: {
        title: 'MultiMind',
        logo: {
          alt: 'MultiMind SDK',
          src: '../assets/Logo-with-name-final2.png',
          width: 160,
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Documentation',
          },
          {
            to: '/docs/features',
            label: 'Features',
            position: 'left',
          },
          {
            to: '/docs/api',
            label: 'API Reference',
            position: 'left',
          },
          {
            to: '/docs/architecture',
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
                to: '/docs/api',
              },
              {
                label: 'Architecture',
                to: '/docs/architecture',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Discord',
                href: 'https://discord.gg/K64U65je7h',
              },
              {
                label: 'Twitter',
                href: 'x.com/Ai2Innovate',
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
                href: 'https://www.multimind.dev/blogs',
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