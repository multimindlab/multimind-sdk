/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/quickstart',
        'getting-started/configuration',
      ],
    },
    {
      type: 'category',
      label: 'Features',
      items: [
        'features/core-features',
        'features/advanced-features',
        'features/implementation-status',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/overview',
        'architecture/components',
        'architecture/flows',
        'architecture/deployment',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/rag-api',
        'api/client-library',
        'api/authentication',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/basic-usage',
        'guides/advanced-usage',
        'guides/best-practices',
      ],
    },
  ],
};

module.exports = sidebars; 