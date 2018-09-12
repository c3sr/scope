var Renderer = require("docsify-server-renderer");
var readFileSync = require("fs").readFileSync;
var writeFileSync = require("fs").writeFileSync;

const middleware = [];

const template = readFileSync("./index.html", "utf-8")
  .replace("NETLIFY_REPOSITORY_URL", process.env.REPOSITORY_URL)
  .replace("NETLFIY_BRANCH", process.env.BRANCH)
  .replace("NETLFIY_PULL_REQUEST", process.env.PULL_REQUEST)
  .replace("NETLFIY_HEAD", process.env.HEAD)
  .replace("NETLFIY_COMMIT_REF", process.env.COMMIT_REF)
  .replace("NETLFIY_CONTEXT", process.env.CONTEXT)
  .replace("NETLFIY_REVIEW_ID", process.env.REVIEW_ID)
  .replace("NETLIFY_URL", process.env.URL)
  .replace("NETLIFY_DEPLOY_URL", process.env.DEPLOY_URL)
  .replace("NETLIFY_DEPLOY_PRIME_URL", process.env.DEPLOY_PRIME_URL);

writeFileSync("index.html", template);

