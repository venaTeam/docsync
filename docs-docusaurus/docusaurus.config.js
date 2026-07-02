// @ts-check
// Minimal Docusaurus (classic preset) scaffold. docsync authors the `docs/` pages and
// maintains `sidebars.js`; this file is the framework config a real `npx create-docusaurus`
// would emit. Run `npm install && npm run build` to render the site.

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'docsync',
  tagline: 'Keep documentation in sync with code, across repos',
  url: 'https://docsync.example.com',
  baseUrl: '/',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: 'docs',
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
        },
        blog: false,
        theme: {},
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'docsync',
        items: [],
      },
    }),
};

module.exports = config;
