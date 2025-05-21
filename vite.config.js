import { defineConfig } from "vite";
import path from "path";
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
    build: {
        outDir: path.resolve(__dirname, "./src/webui/static"),
        emptyOutDir: false, // so Vite doesn't clear everything in static
        rollupOptions: {
            input: path.resolve(__dirname, "./src/webui/js/main.js"),
            output: {
                entryFileNames: "bundle.js",
                assetFileNames: "[name].[ext]",
                chunkFileNames: "bundle.js", // optional: force chunks to join
                format: "es",
            },
        },
        sourcemap: true,
        minify: "esbuild",
    },

    plugins: [
        tailwindcss(),
    ],

    resolve: {
        alias: {
            three: path.resolve(
                __dirname,
                "./node_modules/three/build/three.module.js",
            ),
            OrbitControls: path.resolve(
                __dirname,
                "./node_modules/three/examples/jsm/controls/OrbitControls.js",
            ),
            GLTFLoader: path.resolve(
                __dirname,
                "./node_modules/three/examples/jsm/loaders/GLTFLoader.js",
            ),
        },
        extensions: [".js"],
    }
});
