// SimpleLightBox
$(document).ready(function() {
    // Overflow scrollbox
    $('div.arithmatex').each(function() { 
        $(this).wrap('<div class="overflow"></div>'); 
    });
    // SimpleLightBox
    var productImageGroups = [];
    $('.img-fluid').each(function() { 
        var productImageSource = $(this).attr('src');
        var productImageTag = $(this).attr('tag');
        var productImageTitle = $(this).attr('title');
        if ( productImageTitle != undefined ){
            productImageTitle = 'title="' + productImageTitle + '" ';
        }
        else {
            productImageTitle = '';
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
$('pre.mermaid-container').each(function() {
    var block = $(this).children('code');
    block.replaceWith('<div class="mermaid" >' + block.text() +'</div>')
    $(this).children('button').remove();
});

mermaid.initialize({
    startOnLoad: true,
    theme: 'default',
    themeCSS: '.label { font-family: "Noto Sans", "Helvetica Neue", Helvetica, Arial, "Noto Serif SC", sans-serif; }',
    mermaid: {
        callback: function (id) {
            $('#' + id).parent().parent().addClass('show');
        },
    },
});

// MathJax
window.MathJax = {
    loader: {
        load: ['[tex]/boldsymbol', '[tex]/color']
    },
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        tags: 'ams',
        displayMath: [ ["\\[","\\]"] ],
        tagSide: "right",
        tagIndent: ".8em",
        multlineWidth: "85%",
        packages: {'[+]': ['base', 'boldsymbol', 'color']},
        processEscapes: true,
        processEnvironments: true
    },
    chtml: {
        displayAlign: "center"
    },
    options: {
        ignoreHtmlClass: '.*|',
        processHtmlClass: 'arithmatex'
    },
    svg: {
        fontCache: 'global'
    }
};

(function () {
    var script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.1.2/es5/tex-mml-chtml.js';
    script.async = true;
    document.head.appendChild(script);
})();

// jax: ["input/TeX","output/HTML-CSS"],
//         tex: {
//             tags: 'ams'
//         },
//         tex2jax: {
//             inlineMath: [ ["\\(","\\)"] ],
//             displayMath: [ ["\\[","\\]"] ]
//         },
//         TeX: {
//             TagSide: "right",
//             TagIndent: ".8em",
//             MultLineWidth: "85%",
//             equationNumbers: {
//                 autoNumber: "AMS",
//             },
//             extensions: ["boldsymbol.js", "color.js"],
//             unicode: {
//                 fonts: "STIXGeneral,'Arial Unicode MS'"
//             }
//         },
//         displayAlign: "center",
//         showProcessingMessages: false,
//         messageStyle: "none",