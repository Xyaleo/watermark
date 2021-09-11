//提示框
function prompt(message, style, tip) {
    if ($(".alert").length > 0) {
        $(".alert").remove();
    }
    style = (style === undefined) ? 'alert-success' : style;
    let oDiv = $('<div>')
        .appendTo('body')
        .addClass('alert ' + style)
        .html('<div style="font-size: 18px">' + tip + '</div>' +
            '<div>' + message + '</div>')
        .show()
        .animate({top: 30}, 500, function () {
            setTimeout(function () {
                oDiv.animate({opacity: 0}, 500, function () {
                    oDiv.remove();
                });
            }, 1000);
        })
}

// 成功提示
function success_prompt(message, tip) {
    prompt(message, 'alert-success', tip);
}

// 失败提示
function fail_prompt(message, tip) {
    prompt(message, 'alert-danger', tip);
}

// 提醒
function warning_prompt(message, tip) {
    prompt(message, 'alert-warning', tip);
}
