
var na = document.getElementById('na');
console.log(na)
window.onscroll = function() {
    if (window.pageYOffset > 100) {
        na.style.background= "rgb(0, 114, 122)";
    } else {
        na.style.background= "rgba(255,255,255,0.2)";
    }
}

