// postcss.config.js
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
    'mini-css-extract-plugin': {},
    'postcss-preset-env': {
      browserslist: ['last 2 versions', 'not dead']
    }
  }
};