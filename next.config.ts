const {
  PHASE_DEVELOPMENT_SERVER,
  PHASE_PRODUCTION_BUILD,
} = require("next/constants");

/** @type {import("next").NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  sassOptions: {
    quietDeps: true,
    silenceDeprecations: [
      "import",
      "color-functions",
      "global-builtin",
      "legacy-js-api",
    ],
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      // ✅ Tell Next.js not to bundle tfjs-node (server-only native module)
      config.externals.push("@tensorflow/tfjs-node");
    }
    return config;
  },
};

module.exports = (phase) => {
  if (phase === PHASE_PRODUCTION_BUILD) {
    const withPWA = require("@ducanh2912/next-pwa").default({
      dest: "public",
      register: true,
    });
    return withPWA(nextConfig);
  }
  return nextConfig;
};
