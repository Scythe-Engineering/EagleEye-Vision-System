import { defineConfig } from "vite";
import path from "path";
import tailwindcss from '@tailwindcss/vite';
import handlebars from 'vite-plugin-handlebars';

export default defineConfig({
    root: path.resolve(__dirname, "./src/webui"),
    build: {
        outDir: path.resolve(__dirname, "./src/webui/static"),
        emptyOutDir: true,
        rollupOptions: {
            input: {
                main: path.resolve(__dirname, "./src/webui/js/main.js"),
                index: path.resolve(__dirname, "./src/webui/index.html"),
            },
            output: {
                entryFileNames: "bundle.js",
                assetFileNames: "[name].[ext]",
                chunkFileNames: "bundle.js", // optional: force chunks to join
                format: "es",
            },
        },
        sourcemap: false,
        minify: "esbuild",
    },

    plugins: [
        tailwindcss(),
        handlebars({
            partialDirectory: [
                path.resolve(__dirname, "./src/webui/html/tabs"),
                path.resolve(__dirname, "./src/webui/html/partials"),
            ],
        }),
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
    },

    esbuild: {
        minify: true,
        legalComments: 'none',
    },
});
