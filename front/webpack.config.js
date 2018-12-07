const path = require('path');
const webpack = require('webpack');
const basePath = path.resolve(__dirname);
const ExtractTextPlugin = require('extract-text-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const dev = {
    context: path.join(basePath, 'src'),
    entry: {
        style: ['./styles/main.scss'],
        main: './scripts/main.js'
    },
    module: {
        rules:[
            { test: /\.scss$/, use: ExtractTextPlugin.extract({
                    fallback: 'style-loader',
                    use: ['css-loader', 'sass-loader']
                })
            }, // SASS & CSS FILES
            { test: /\.(jpe?g|png|gif|svg|eot)$/i, exclude: /(node_modules|bower_components)/, loader: 'file-loader?limit=1000&name=/images/[hash].[ext]' }, // IMAGES
            {test: /\.(html)$/,
                use: {
                    loader: 'html-loader',
                    options: {
                        attrs: false,//['img:src', 'link:href'],
                        minimize: true,
                        removeComments: true,
                        collapseWhitespace: true

                    }
                }
            }
        ]
    },
    output: {
        path: path.join(basePath, 'static'),
        filename: '[name].min.js',
        publicPath: ''
    },
    devServer: {
        contentBase: path.join(basePath, 'static'),
        watchContentBase: true,
        proxy: {
            '/v1': process.env.ENV_API || 'http://localhost:8000',
            '/data': process.env.ENV_API || 'http://localhost:8000'
        }
    },
    resolve: {
        alias: {
            videojs: 'video.js/dist/video.es.js',
            WaveSurfer: 'wavesurfer.js',
            jQuery: 'jquery',
            $: 'jquery',
            jquery: 'jquery'
        }
    },
    plugins: [
        new webpack.ProvidePlugin({
            videojs: 'video.js/dist/video.cjs.js',
            RecordRTC: 'recordrtc',
            'window.videojs': 'video.js',
            'window.RecordRTC': 'recordrtc',
            jQuery: 'jquery',
            $: 'jquery',
            jquery: 'jquery',
            MediaStreamRecorder: ['recordrtc', 'MediaStreamRecorder']
        }),
        new ExtractTextPlugin('main.css', {
            allChunks: true
        }),
        new HtmlWebpackPlugin({
            title: 'Fake detector',
            template: 'index.html',
            inject:true,
            hash:true,
            "files": {
                "css": [ "main.css?v=0.1" ],
                "js": [ "style.min.js?v=0.1", "main.min.js?v=0.1"],
                "chunks": {
                  "head": {
                    "entry": "style.min.js?v=0.1",
                    "css": [ "main.css?v=0.1" ]
                  },
                  "main": {
                    "entry": "main.min.js?v=0.1",
                    "css": []
                  },
                }
            }
        }),
        new CopyWebpackPlugin([
            {
              from: path.join(basePath, 'src/images'),
              to: path.join(basePath, 'static/images'),
              cache: false
            }
        ], {})
    ]
};


const production = {
    context: path.join(basePath, 'src'),
    entry: {
        style: ['./styles/main.scss'],
        main: './scripts/main.js'
    },
    module: {
        rules:[
            { test: /\.scss$/, use: ExtractTextPlugin.extract({
                    fallback: 'style-loader',
                    use: ['css-loader', 'sass-loader']
                })
            }, // SASS & CSS FILES
            { test: /\.(jpe?g|png|gif|svg|eot)$/i, exclude: /(node_modules|bower_components)/, loader: 'file-loader?limit=1000&name=/images/[hash].[ext]' }, // IMAGES
            {test: /\.(html)$/,
                use: {
                    loader: 'html-loader',
                    options: {
                        attrs: ['img:src', 'link:href'],
                        minimize: true,
                        removeComments: true,
                        collapseWhitespace: true

                    }
                }
            }
        ]
    },
    output: {
        path: path.join(basePath, 'static'),
        filename: '[name].min.js',
        publicPath: ''
    },
    devServer: {
        contentBase: path.join(basePath, 'static'),
        watchContentBase: true,
        proxy: {
            '/v1': process.env.ENV_API || 'http://localhost:8000'
        }
    },
    resolve: {
        alias: {
            videojs: 'video.js/dist/video.es.js',
            WaveSurfer: 'wavesurfer.js',
            jQuery: 'jquery',
            $: 'jquery',
            jquery: 'jquery'
        }
    },
    plugins: [
        new webpack.ProvidePlugin({
            videojs: 'video.js/dist/video.cjs.js',
            RecordRTC: 'recordrtc',
            'window.videojs': 'video.js',
            'window.RecordRTC': 'recordrtc',
            jQuery: 'jquery',
            $: 'jquery',
            jquery: 'jquery',
            MediaStreamRecorder: ['recordrtc', 'MediaStreamRecorder']
        }),
        new ExtractTextPlugin('main.css', {
            allChunks: true
        }),
        new HtmlWebpackPlugin({
            title: 'Fake detector',
            template: 'index_2.html',
            inject:true,
            hash:true,
            "files": {
                "css": [ "main.css?v=0.1" ],
                "js": [ "style.min.js?v=0.1", "main.min.js?v=0.1"],
                "chunks": {
                  "head": {
                    "entry": "style.min.js?v=0.1",
                    "css": [ "main.css?v=0.1" ]
                  },
                  "main": {
                    "entry": "main.min.js?v=0.1",
                    "css": []
                  },
                }
            }
        })
    ]
};


module.exports =(env) => {
    if(env == "prod")
        return production
    else{
        return dev
    }
}