// SimpleLightBox
$(document).ready(function() {
    // Overflow scrollbox
    $('div .overflow').each(function() { 
        $(this).wrapInner('<div class="check" />'); 
    });
    // SimpleLightBox
    var productImageGroups = [];
    $('.img-fluid').each(function() { 
        var productImageSource = $(this).attr('src');
        var productImageTag = $(this).attr('tag');
        var productImageTitle = $(this).attr('title');
        if ( productImageTitle != undefined ){
            productImageTitle = 'title="' + productImageTitle + '" '
        }
        else {
            productImageTitle = ''
        }
        $(this).wrap('<a class="boxedThumb ' + productImageTag + '" ' + productImageTitle + 'href="' + productImageSource + '"></a>');
        productImageGroups.push('.'+productImageTag);
    });
    jQuery.unique( productImageGroups );
    productImageGroups.forEach(productImageGroupsSet);
    function productImageGroupsSet(value) {
        $(value).simpleLightbox();
    }
});

// Mermaid
mermaid.initialize({
  startOnLoad: true,
  theme: 'neutral',
  themeCSS: '.label { font-family: "Helvetica Neue", Helvetica, Arial, "Noto Serif SC", sans-serif; }'
});

// MathJax
window.MathJax = {
    jax: ["input/TeX","output/HTML-CSS"],
    tex2jax: {
      inlineMath: [ ["\\(","\\)"] ],
      displayMath: [ ["\\[","\\]"] ]
    },
    TeX: {
      TagSide: "right",
      TagIndent: ".8em",
      MultLineWidth: "85%",
      equationNumbers: {
        autoNumber: "AMS",
      },
      extensions: ["boldsymbol.js", "color.js"],
      unicode: {
        fonts: "STIXGeneral,'Arial Unicode MS'"
      }
    },
    displayAlign: "center",
    showProcessingMessages: false,
    messageStyle: "none",
  };