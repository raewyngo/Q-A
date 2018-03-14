$(document).ready(function(){
	$("#search").click(function(url){
		inputquestion = $("#inputquestion").val();
		inputarticle = $("#inputarticle").val();

		$("#loading").show();
		$("#results").hide();
		$.ajax({
			url:"/answer/",
			type:"post",
			data: {"inputarticle":inputarticle, "inputquestion": inputquestion},
			dataType:"",
			success:function(data){
				if (data == "ERROR") {
					alert("查找失败")
				}
				else {
					$("#results").html(data);
				}
				$("#loading").hide();
				$("#results").show();
			},
			error:function(url){
				alert("请求失败");
				$("#loading").hide();
				$("#results").hide();
			},
		});
	});

});

