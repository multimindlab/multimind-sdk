const fs = require('fs');
const path = require('path');

const DOCS_SOURCE = path.join(__dirname, '../docs');
const DOCS_TARGET = path.join(__dirname, '../multimind-docs/docs');

// Create necessary directories
const dirs = [
  'getting-started',
  'features',
  'architecture',
  'api',
  'guides'
];

dirs.forEach(dir => {
  fs.mkdirSync(path.join(DOCS_TARGET, dir), { recursive: true });
});

// Migration mappings
const migrations = [
  {
    source: 'README.md',
    target: 'intro.md',
    transform: (content) => {
      // Add frontmatter
      return `---
id: intro
title: Introduction
sidebar_position: 1
---

${content}`;
    }
  },
  {
    source: 'features.md',
    target: 'features/core-features.md',
    transform: (content) => {
      return `---
id: core-features
title: Core Features
sidebar_position: 1
---

${content}`;
    }
  },
  {
    source: 'architecture.md',
    target: 'architecture/overview.md',
    transform: (content) => {
      return `---
id: overview
title: Architecture Overview
sidebar_position: 1
---

${content}`;
    }
  },
  {
    source: 'api_reference/rag_api.md',
    target: 'api/rag-api.md',
    transform: (content) => {
      return `---
id: rag-api
title: RAG API Reference
sidebar_position: 1
---

${content}`;
    }
  }
];

// Migrate files
migrations.forEach(({ source, target, transform }) => {
  const sourcePath = path.join(DOCS_SOURCE, source);
  const targetPath = path.join(DOCS_TARGET, target);

  if (fs.existsSync(sourcePath)) {
    const content = fs.readFileSync(sourcePath, 'utf8');
    const transformedContent = transform(content);
    fs.writeFileSync(targetPath, transformedContent);
    console.log(`Migrated ${source} to ${target}`);
  } else {
    console.warn(`Source file not found: ${source}`);
  }
});

// Create placeholder files for missing documentation
const placeholders = [
  {
    path: 'getting-started/installation.md',
    title: 'Installation',
    content: `---
id: installation
title: Installation
sidebar_position: 1
---

# Installation

Coming soon...`
  },
  {
    path: 'getting-started/quickstart.md',
    title: 'Quickstart',
    content: `---
id: quickstart
title: Quickstart Guide
sidebar_position: 2
---

# Quickstart Guide

Coming soon...`
  },
  {
    path: 'guides/basic-usage.md',
    title: 'Basic Usage',
    content: `---
id: basic-usage
title: Basic Usage Guide
sidebar_position: 1
---

# Basic Usage Guide

Coming soon...`
  }
];

placeholders.forEach(({ path: filePath, title, content }) => {
  const fullPath = path.join(DOCS_TARGET, filePath);
  if (!fs.existsSync(fullPath)) {
    fs.writeFileSync(fullPath, content);
    console.log(`Created placeholder: ${filePath}`);
  }
});

console.log('Documentation migration completed!'); 