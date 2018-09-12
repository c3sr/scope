var occa = occa || {};

occa.languageLabels = {
  cpp: "C++",
  cuda: "cuda",
  python: "python",
};

occa.getLanguageLabel = language =>
  occa.languageLabels[language] || occa.languageLabels[language.toUpperCase()];

//---[ Header & Footer ]----------------
occa.addFooter = content =>
  content +
  "\n\n" +
  '<div id="footer">\n' +
  "  <hr/>\n" +
  '  <div id="copyright">\n' +
  `   © Copyright 2018 - ${new Date().getFullYear()}, Abdul Dakkak and Cheng Li. &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; Last modified on {docsify-updated} \n` +
  "  </div>\n" +
  "</div>\n";
//======================================

//---[ Indent ]-------------------------
occa.parseIndent = content => {
  const parts = marked.lexer(content);
  const mdContent = occa.tokensToHTML(parts);
  return '<div class="indent">\n' + mdContent + "</div>\n";
};

occa.addIndents = content => {
  const re = /\n::: indent\n([\s\S]*?)\n:::(\n|$)/g;
  const parts = [];
  var lastIndex = 0;
  let match;
  while ((match = re.exec(content)) != null) {
    const [fullMatch, indentContent] = match;

    parts.push(content.substring(lastIndex, match.index));
    parts.push(occa.parseIndent(indentContent));
    lastIndex = match.index + fullMatch.length;
  }
  parts.push(content.substring(lastIndex));

  return parts.join("\n");
};
//======================================

//---[ Math ]-------------------------
occa.parseMath = content => {
  try {
    const html = katex.renderToString(content);
    return html.replace(
      /class="katex"/g,
      'class="katex katex-block" style="display: block;"'
    );
  } catch (e) {
    console.error(e);
    return '<div class="math">\n' + content + "</div>\n";
  }
};

occa.addMaths = content => {
  const re = /\n::: math\n([\s\S]*?)\n:::(\n|$)/g;
  const parts = [];
  var lastIndex = 0;
  let match;
  while ((match = re.exec(content)) != null) {
    const [fullMatch, indentContent] = match;

    parts.push(content.substring(lastIndex, match.index));
    parts.push(occa.parseMath(indentContent));
    lastIndex = match.index + fullMatch.length;
  }
  parts.push(content.substring(lastIndex));

  return parts.join("\n");
};
//======================================

//---[ Tabs ]---------------------------
occa.markdown = {
  space: () => "",
  text: ({ text }) => `<p>${text}</p>`,
  paragraph: ({ text }) => `<p>${text}</p>`,
  list_start: () => "<ul>",
  list_end: () => "</ul>",
  list_item_start: () => "<li>",
  list_item_end: () => "</li>",
  html: ({ text }) => text
};

occa.markdown.code = ({ lang, text }) => {
  // Remove indentation
  const initIndent = text.match(/^\s*/)[0];
  if (initIndent.length) {
    const lines = text.split(/\r?\n/);
    const isIndented = lines.every(
      line => !line.length || line.startsWith(initIndent)
    );

    if (isIndented) {
      text = lines.map(line => line.substring(initIndent.length)).join("\n");
    }
  }

  // Generate highlighted HTML
  const styledCode = Prism.highlight(text, Prism.languages[lang], lang);

  // Wrap around pre + code
  return (
    `<pre v-pre data-lang="${occa.getLanguageLabel(lang)}">` +
    `<code class="lang-${lang}">` +
    `${styledCode}\n` +
    "</code>" +
    "</pre>"
  )
    .replace(/[*]/g, "&#42;")
    .replace(/[_]/g, "&#95;");
};

occa.tokenToMarkdown = token => {
  const { type } = token;
  if (type in occa.markdown) {
    return occa.markdown[token.type](token);
  }
  console.error(`Missing token format for: ${token.type}`, token);
  return "";
};

occa.mergeTextTokens = tokens => {
  const newTokens = [];
  let texts = [];
  for (var i = 0; i < tokens.length; ++i) {
    const token = tokens[i];
    if (token.type === "text") {
      texts.push(token.text);
      continue;
    }
    if (texts.length) {
      newTokens.push({
        type: "text",
        text: texts.join(" ")
      });
      texts = [];
    }
    newTokens.push(token);
  }
  // Join the tail texts
  if (texts.length) {
    newTokens.push({
      type: "text",
      text: texts.join(" ")
    });
  }
  return newTokens;
};

occa.tokensToHTML = tokens => {
  tokens = occa.mergeTextTokens(tokens);
  return tokens.map(occa.tokenToMarkdown).join("\n");
};

occa.getTab = ({ tab, content }) =>
  `      <md-tab id="${tab}" md-label="${tab}">\n` +
  occa.tokensToHTML(content) +
  "      </md-tab>";

occa.getTabs = (namespace, tabs) => {
  const content = tabs.map(occa.getTab).join("\n");

  const tab = `vm.$data.tabNamespaces['${namespace}']`;
  const onClick = `(tab) => vm.onTabChange('${namespace}', tab)`;

  return (
    "<template>\n" +
    "  <div>\n" +
    "    <md-tabs\n" +
    '      md-dynamic-height="true"\n' +
    `      v-bind:md-active-tab="${tab}"\n` +
    `      @md-changed="${onClick}"\n` +
    "    >\n" +
    `${content}\n` +
    "    </md-tabs>\n" +
    "  </div>\n" +
    "</template>\n"
  );
};

occa.parseTabs = (namespace, content) => {
  const parts = marked.lexer(content);
  const newParts = [];

  // Skip begin/end of list
  for (var i = 1; i < parts.length - 1; ++i) {
    var stackSize = 1;

    // Skip loose_item_start;
    ++i;
    const tab = parts[i++].text;
    const start = i++;

    while (i < parts.length - 1 && stackSize > 0) {
      switch (parts[i].type) {
        case "list_item_start":
          ++stackSize;
          break;
        case "list_item_end":
          --stackSize;
          break;
      }
      ++i;
    }

    // Don't take the token after list_item_end
    --i;

    newParts.push({
      tab,
      content: parts.slice(start, i)
    });
  }

  if (!newParts.length) {
    return [];
  }

  if (!(namespace in vm.$data.tabNamespaces)) {
    Vue.set(vm.tabNamespaces, namespace, newParts[0].tab);
  }

  return occa.getTabs(namespace, newParts);
};

occa.addTabs = content => {
  const re = /\n::: tabs (.*)\n([\s\S]*?)\n:::(\n|$)/g;
  const parts = [];
  var lastIndex = 0;
  while ((match = re.exec(content)) != null) {
    const [fullMatch, namespace, tabContent] = match;

    parts.push(content.substring(lastIndex, match.index));
    parts.push(occa.parseTabs(namespace, tabContent));

    lastIndex = match.index + fullMatch.length;
  }
  parts.push(content.substring(lastIndex));

  return parts.join("\n");
};

//======================================

// Root-level markdowns don't have a sidebar
occa.hasSidebar = file => !file.match(/^[^/]*\.md$/);

occa.docsifyPlugin = (hook, vm) => {
  hook.init(() => {
    Prism.languages.sh = Prism.languages.extend("sh");
    Prism.languages.bibtex = Prism.languages.extend("latex");
  });

  hook.beforeEach(content => {
    // No \n means the last line turns into a header
    if (!content.endsWith("\n")) {
      content += "\n";
    }
    content = occa.addMaths(content);
    content = occa.addIndents(content);
    content = occa.addTabs(content);
    content = occa.addFooter(content);
    return content;
  });

  hook.doneEach(() => {
    const dom = document.querySelector("body");
    const file = vm.route.file;
    // Add Benchmark styling
    if (!file.startsWith("benchmarks/")) {
      dom.classList.remove("benchmark-container");
    } else {
      dom.classList.add("benchmark-container");
    }
    // Close sidebar
    if (occa.hasSidebar(file)) {
      dom.classList.remove("no-sidebar");
    } else {
      dom.classList.add("no-sidebar");
    }
  });
};
