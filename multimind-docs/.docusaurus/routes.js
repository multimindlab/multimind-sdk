import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/search',
    component: ComponentCreator('/search', '5de'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '1ab'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '407'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', 'f98'),
            routes: [
              {
                path: '/docs/api/',
                component: ComponentCreator('/docs/api/', '2c7'),
                exact: true
              },
              {
                path: '/docs/api/authentication',
                component: ComponentCreator('/docs/api/authentication', '2bd'),
                exact: true
              },
              {
                path: '/docs/api/client-library',
                component: ComponentCreator('/docs/api/client-library', '4ed'),
                exact: true
              },
              {
                path: '/docs/api/rag-api',
                component: ComponentCreator('/docs/api/rag-api', '116'),
                exact: true
              },
              {
                path: '/docs/architecture',
                component: ComponentCreator('/docs/architecture', '4b2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/architecture/',
                component: ComponentCreator('/docs/architecture/', '3a6'),
                exact: true
              },
              {
                path: '/docs/architecture/overview',
                component: ComponentCreator('/docs/architecture/overview', '685'),
                exact: true
              },
              {
                path: '/docs/contributing',
                component: ComponentCreator('/docs/contributing', '069'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/examples/basic-agent',
                component: ComponentCreator('/docs/examples/basic-agent', 'c43'),
                exact: true
              },
              {
                path: '/docs/features/',
                component: ComponentCreator('/docs/features/', '749'),
                exact: true
              },
              {
                path: '/docs/features/advanced-features',
                component: ComponentCreator('/docs/features/advanced-features', 'cc0'),
                exact: true
              },
              {
                path: '/docs/features/core-features',
                component: ComponentCreator('/docs/features/core-features', '210'),
                exact: true
              },
              {
                path: '/docs/features/implementation-status',
                component: ComponentCreator('/docs/features/implementation-status', '5d0'),
                exact: true
              },
              {
                path: '/docs/getting-started',
                component: ComponentCreator('/docs/getting-started', '2a1'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/',
                component: ComponentCreator('/docs/getting-started/', '900'),
                exact: true
              },
              {
                path: '/docs/getting-started/installation',
                component: ComponentCreator('/docs/getting-started/installation', '2f7'),
                exact: true
              },
              {
                path: '/docs/getting-started/quickstart',
                component: ComponentCreator('/docs/getting-started/quickstart', '748'),
                exact: true
              },
              {
                path: '/docs/guides/basic-usage',
                component: ComponentCreator('/docs/guides/basic-usage', '583'),
                exact: true
              },
              {
                path: '/docs/integration-guide',
                component: ComponentCreator('/docs/integration-guide', '098'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '853'),
                exact: true
              },
              {
                path: '/docs/introduction',
                component: ComponentCreator('/docs/introduction', 'f7d'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
