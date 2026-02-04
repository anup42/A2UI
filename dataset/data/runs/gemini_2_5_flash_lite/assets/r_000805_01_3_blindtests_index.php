
<!DOCTYPE html>
<HTML lang="en">
<head>
	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<meta name="robots" content="index, follow">
<meta name="viewport" content="width=device-width, initial-scale=1">


<link rel="image_src" href="/Pix/FBdefault.jpg">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:site" content="@audiosampling">
<meta name="twitter:title" content="Blind Tests">
<meta name="twitter:description" content="Test your audio equipment online. Check for frequency response, dynamic range, stereo imaging and more!">
<meta name="twitter:image:src" content="https://www.audiocheck.net/Pix/FBdefault.jpg">

<!-- Open Graph -->
<meta property="og:image" content="https://www.audiocheck.net/Pix/FBdefault.jpg" >

<title>Blind Tests</title>
<meta name="description" content="Test your audio equipment online. Check for frequency response, dynamic range, stereo imaging and more!">
<meta name="author" content="Dr. Ir. St&eacute;phane Pigeon">
<meta name="keywords" content="audio, audio tests, audio test files, test tone, test tones, audio test tones, download, free download, audio test signals, audio testing, audio testing tools, audio benchmarking, online audio tests, speaker test, speaker testing, blind tests, white noise, pink noise, dynamic range, brown noise, audio generator, sound, sound tests, sound test files, sound test signals, sound testing, sound testing tools, sound benchmarking, online sound tests, sound generator, waveform, waveform generator, function generator, sine tone, pure tone, hearing">	
	<script src="//ajax.googleapis.com/ajax/libs/jquery/2.2.4/jquery.min.js"></script>
	
	<!-- icons -->
	<link rel="image_src" href="./IMG/ear.png" /> 
	<link rel="apple-touch-icon" sizes="57x57" href="/apple-touch-icon-57x57.png">
	<link rel="apple-touch-icon" sizes="60x60" href="/apple-touch-icon-60x60.png">
	<link rel="apple-touch-icon" sizes="72x72" href="/apple-touch-icon-72x72.png">
	<link rel="apple-touch-icon" sizes="76x76" href="/apple-touch-icon-76x76.png">
	<link rel="apple-touch-icon" sizes="114x114" href="/apple-touch-icon-114x114.png">
	<link rel="apple-touch-icon" sizes="120x120" href="/apple-touch-icon-120x120.png">
	<link rel="apple-touch-icon" sizes="144x144" href="/apple-touch-icon-144x144.png">
	<link rel="apple-touch-icon" sizes="152x152" href="/apple-touch-icon-152x152.png">
	<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon-180x180.png">
	<link rel="icon" type="image/png" href="/favicon-32x32.png" sizes="32x32">
	<link rel="icon" type="image/png" href="/favicon-194x194.png" sizes="194x194">
	<link rel="icon" type="image/png" href="/favicon-96x96.png" sizes="96x96">
	<link rel="icon" type="image/png" href="/android-chrome-192x192.png" sizes="192x192">
	<link rel="icon" type="image/png" href="/favicon-16x16.png" sizes="16x16">
	<link rel="manifest" href="/manifest.json">
	<meta name="msapplication-TileColor" content="#da532c">
	<meta name="msapplication-TileImage" content="/mstile-144x144.png">
	<meta name="theme-color" content="#ffffff">

	<!-- Override CSS file - add your own CSS rules -->
	<link rel="stylesheet" href="styles.css">
		
	<!-- Google Fonts -->
	<link href='https://fonts.googleapis.com/css?family=Abel|Inconsolata' rel='stylesheet' type='text/css'>
	
	<!-- JS Code -->
	<script>

// Read Cookies
function getCookieValue(a, b) {
    b = document.cookie.match('(^|;)\\s*' + a + '\\s*=\\s*([^;]+)');
    return b ? b.pop() : '';
}

		
var fileExt=".mp3";
var fileNameArray=new Array();
var audioArray=new Array();
var allAudioButtons=new Array();
var allSoundIDs=new Array();
var currentDraw;
var blindAudio=new Audio();
var trials=0;
var score=0;
var pp=0;
			
function init(){
	setContentsPosition();
	checkAudioFileCompatibility();
	goThroughAllAudioButtons();
	collectSoundIDs();
	window.onresize = function() {setContentsPosition();};
	pp=getCookieValue('patreonPledge');
}

function setContentsPosition(){
	var height = document.getElementById("onTop").offsetHeight;
	document.getElementById("contents").style.marginTop = height + 'px';
	document.getElementById("contents").style.visibility = 'visible';
}
		
function checkAudioFileCompatibility(){
	var a = document.createElement('audio');
	if (!!(a.canPlayType && a.canPlayType('audio/ogg; codecs="vorbis"').replace(/no/, ''))) fileExt=".ogg";
}
	
function goThroughAllAudioButtons(){
	allAudioButtons = document.getElementsByClassName('audioButton');
	for (var k = 0; k<allAudioButtons.length; k++) {
    	allAudioButtons[k].title="Play audio (click to play/pause)";
	}
}

function collectSoundIDs(){
 for (var k=0;k<allAudioButtons.length-1;k++) {
		allSoundIDs[k]=allAudioButtons[k].id.substr(0,allAudioButtons[k].id.length-3);
		allSoundIDs[k]=decodeURIComponent(allSoundIDs[k]); // plus signs in file names gave trouble, as they were encoded
	}
 blindDraw();
}

function blindDraw(){

	var xS=document.getElementById('xSound');
	if (xS) {
		var r=Math.floor(Math.random() * allSoundIDs.length);
		currentDraw=allSoundIDs[r];
		if (pp>150) console.log('Current Draw : '+currentDraw);
		xS.className='audioButtonIsBlind';
	}
	

}

function playBlind(){

	var xS=document.getElementById('xSound');
	if (xS) {
		//xS.className='audioButtonIsLoading';
		//xS.onclick='';
		//xS.title="Please wait (audio file is loading)";
		
		blindAudio.src='/Audio/'+currentDraw+fileExt;
		blindAudio.oncanplay=function(){
			xS.className='audioButtonIsPlaying'; 
			xS.onclick=function(){playBlind();}; 
			xS.title="Play audio (click to play/pause)";
		};
		blindAudio.onended=function(){xS.className='audioButtonIsBlind'};
		blindAudio.onpause=function(){xS.className='audioButtonIsBlind'};
		blindAudio.loop=0;
		blindAudio.volume=1;
		//mute all currently playing
		for (var k=0;k<audioArray.length;k++) {
			if (!isNaN(audioArray[k].duration)) audioArray[k].pause();
		}
		blindAudio.play();
		document.getElementById('resultTxt').innerHTML='Trial '+(trials+1);
		}
}

var f = [];
function fac (n) {
  if (n == 0 || n == 1)
    return 1;
  if (f[n] > 0)
    return f[n];
  return f[n] = fac(n-1) * n;
}

function vote(file){

		blindAudio.pause();

		trials++;		
		var txt='';
		if (file==currentDraw) { 
			score++;
			txt+='<span class="green">Correct </span> — ';
		} else txt+='<span class="red">Wrong </span> — ';
		var p=1/allSoundIDs.length;
		var sig =1;
					
		for (var k=score;k<=trials;k++) sig-=(fac(trials)/(fac(k)*fac(trials-k)))*Math.pow(p,k)*Math.pow(1-p,trials-k);
		
		sig=Math.max(0,Math.floor(10000*sig)/100);
		var percentage=Math.round(100*score/trials);
	
		txt+='Current score: '+score+'/'+trials+' ('+percentage+'%) — ';	
		if (sig>=95) txt+='<span class="green">Confidence : '+sig+'%</span>';
			else txt+='Confidence : '+sig+'%';
			
if ((trials>9)&&(sig>=95)&&(percentage>80))	txt+='<br>&#128077; It feels like you have successfully passed the test!</span>';
		document.getElementById('resultTxt').innerHTML=txt;
		blindDraw();
}
		
function download(audiocheckSoundFile){
   window.location= './download.php?filename=Audio/'+audiocheckSoundFile+'.wav';
}

function play(audiocheckSoundFileWithUID){
var audiocheckSoundFile=audiocheckSoundFileWithUID.substr(0,audiocheckSoundFileWithUID.length-3);
var i=fileNameArray.indexOf(audiocheckSoundFileWithUID);
var clicked=document.getElementById(audiocheckSoundFileWithUID);

// mute all others
for (var k=0;k<audioArray.length;k++) {
		if (k!=i && !isNaN(audioArray[k].duration)) audioArray[k].pause();
	}
if (!isNaN(blindAudio.duration)) {blindAudio.pause();}
	
// play clicked
if (i==-1){ // never played before
	clicked.className='audioButtonIsLoading';
	clicked.onclick=''; 
	clicked.title="Please wait (audio file is loading)";
	fileNameArray.push(audiocheckSoundFileWithUID);
	i=fileNameArray.indexOf(audiocheckSoundFileWithUID);
	audioArray[i]=new Audio('/Audio/'+audiocheckSoundFile+fileExt);
	audioArray[i].oncanplay=function(){clicked.className='audioButtonIsPlaying'; clicked.onclick=function(){play(this.id);}; clicked.title="Play audio (click to play/pause)";};
	audioArray[i].onpause=function(){clicked.className='audioButtonIsPaused'};	
	audioArray[i].onended=function(){clicked.className='audioButton'};
	audioArray[i].loop=0;
	audioArray[i].volume=1;
	audioArray[i].play();
}
else { // has played or is playing.. or was loading
	if (audioArray[i].readyState==0) {
		clicked.className="audioButtonIsLoading";
		clicked.title="Please wait (audio file is loading)";
	}
	else {
		audioArray[i].onplay=function(){clicked.className='audioButtonIsPlaying'; clicked.onclick=function(){play(this.id);}; clicked.title="Play audio (click to play/pause)";};
		if (audioArray[i].paused) audioArray[i].play();
		else audioArray[i].pause();
	}
}
}

</script>		
</head>


<BODY onload="init()">
<div id="paymentOverlay"><tt>Your payment is being processed. You will be redirected in a couple of seconds...</tt></div>

<div id="onTop">
	<div id="menu"><div class="menu" id="menu">
<ul>
<li ><a href="./index.php">Home</a></li> &bull;
<li ><a href="./audiotests_index.php">Audio&nbsp;Tests</a></li> &bull;
<li ><a href="./testtones_index.php">Test&nbsp;Tones</a></li> &bull;
<li ><a href="./audiofrequencysignalgenerator_index.php">Tone&nbsp;Gen</a></li> &bull;
<li id="active"><a href="./blindtests_index.php">Blind&nbsp;Tests</a></li>
	</ul>
</div></div>
	<div id="title"><h1>Blind Tests</h1></div>
</div>
<div id="contents">

<div id="colA">

	<div class="section">
		<p class="topSpaced">A "blind test" is a method of testing in which the people being experimented on
have no idea about what they're getting. This test method prevents results
from being influenced by any a priori information. In the field of Audio, blind tests truly highlight what a listener is able to hear.</p>
<p>
In the so-called ABX blind listening test, the listener has access to three sources: A and B are the references, X is the mystery source. X can be A or B.
When the listener says that X is A, and that X is actually A, it doesn't prove anything yet: flipping a coin achieves the same result half of the time anyway. This is why we provide the listener with many trials to determine if the number of correct answers is statistically significant.
</p>
	</div>
	
	<div class="section">
<h2>Take up the challenge</h2> 
<ul>
<li>Find the smallest difference in sound levels you can detect.&nbsp; <br>
The <span class="emphasis">Level</span> Series:&nbsp;
<a href="./blindtests_level.php?lvl=6">6dB</a>&nbsp;
<a href="./blindtests_level.php?lvl=3">3dB</a>&nbsp;
<a href="./blindtests_level.php?lvl=1">1dB</a>&nbsp;
<a href="./blindtests_level.php?lvl=0.5">0.5dB</a>&nbsp;
<a href="./blindtests_level.php?lvl=0.2">0.2dB</a>&nbsp;
<a href="./blindtests_level.php?lvl=0.1">0.1dB</a>&nbsp;
<p>
<li>Find the highest frequency you can reliably hear. <br>
The <span class="emphasis">Frequency</span> Series:&nbsp;
<a href="./blindtests_frequency.php?frq=10">10kHz</a>
<a href="./blindtests_frequency.php?frq=11">11k</a>
<a href="./blindtests_frequency.php?frq=12">12k</a>
<a href="./blindtests_frequency.php?frq=13">13k</a>
<a href="./blindtests_frequency.php?frq=14">14k</a>
<a href="./blindtests_frequency.php?frq=15">15k</a>
<a href="./blindtests_frequency.php?frq=16">16k</a>
<a href="./blindtests_frequency.php?frq=17">17k</a>
<a href="./blindtests_frequency.php?frq=18">18k</a>
<a href="./blindtests_frequency.php?frq=19">19k</a>
<a href="./blindtests_frequency.php?frq=20">20kHz</a>
<p>
<li>Find the smallest difference in pitch (frequency) you can hear.&nbsp; <br>
The <span class="emphasis">Pitch</span> Series:&nbsp;
<a href="./blindtests_pitch.php?cent=50">50c</a>&nbsp;
<a href="./blindtests_pitch.php?cent=20">20c</a>&nbsp;
<a href="./blindtests_pitch.php?cent=10">10c</a>&nbsp;
<a href="./blindtests_pitch.php?cent=5">5c</a>&nbsp;
<a href="./blindtests_pitch.php?cent=2">2c</a>&nbsp;
<a href="./blindtests_pitch.php?cent=1">1c</a>&nbsp;
<p>
<li>Find the shortest timing difference you can reliably hear.  <span class="new">NEW</span> <br>
The <span class="emphasis">Timing</span> Series:&nbsp;
<a href="./blindtests_timing_2w.php?time=1">1ms</a>&nbsp;
<a href="./blindtests_timing_2w.php?time=2">2ms</a>&nbsp;
<a href="./blindtests_timing_2w.php?time=5">5ms</a>&nbsp;
<a href="./blindtests_timing_2w.php?time=10">10ms</a>&nbsp;
<a href="./blindtests_timing_2w.php?time=20">20ms</a>&nbsp;
<a href="./blindtests_timing_2w.php?time=50">50ms</a>&nbsp;
<a href="./blindtests_timing_2w.php?time=100">100ms</a>&nbsp;
<p>
<li>Find the highest dynamic range offered by your listening environment.&nbsp; <br>
The <span class="emphasis">Dynamic Range</span> Series:&nbsp;
<a href="./blindtests_dynamic.php?dyna=36">36dB</a>&nbsp;
<a href="./blindtests_dynamic.php?dyna=42">42dB</a>&nbsp;
<a href="./blindtests_dynamic.php?dyna=48">48dB</a>&nbsp;
<a href="./blindtests_dynamic.php?dyna=54">54dB</a>&nbsp;
<a href="./blindtests_dynamic.php?dyna=60">60dB</a>&nbsp;
<a href="./blindtests_dynamic.php?dyna=66">66dB</a>&nbsp;
<a href="./blindtests_dynamic.php?dyna=72">72dB</a>&nbsp;
<a href="./blindtests_dynamic.php?dyna=78">78dB</a>&nbsp;
<p>
<li>Do you have the absolute hearing ability?&nbsp;<br>
The <span class="emphasis">Perfect Pitch</span> Blind Test:&nbsp;
<a href="./blindtests_abspitch.php">C Scale</a>&nbsp;
<a href="./blindtests_abspitchchromatic.php">Chromatic</a>&nbsp;
<p>
<li>Are your ears sensitive to Absolute Phase?&nbsp; <br>
The <span class="emphasis">Absolute Polarity</span> Blind Test:&nbsp;
<a href="./blindtests_abspolarity.php">Here</a>&nbsp;
<p>
<li>Can you hear a difference between 16-bit and 8-bit audio files?&nbsp; <br>
The
<a href="./blindtests_16vs8bit.php">16-bit v/s 8-bit Blind Test</a>
</ul>
</div>

<div class="section">
<h2>For sound and studio engineers</h2>
<ul>
<li>Learn to discriminate <b>frequencies</b> by ear.<br>
The <span class="emphasis">Peak</span> Series:&nbsp;<a href="./engineertraining_bands_octave.php">10 bands</a>&nbsp;
<a href="./engineertraining_bands_difficult.php">14 bands</a> <br>
The <span class="emphasis">Notch</span> Series:&nbsp;<a href="./engineertraining_dips_octave.php">10 bands</a>&nbsp;
<a href="./engineertraining_dips_difficult.php">14 bands</a><br>
The <span class="emphasis">Music</span> Series: &nbsp;<a href="./engineertraining_music_peaks.php">Peaks</a>&nbsp;
<a href="./engineertraining_music_dips.php">Notches</a>
<li>Check your <a href="./blindtests_timing_3w.php?time=5">timing</a> precision  <span class="new">NEW</span>
</ul>
</div>
</div>				

<div id="colB">

	<div class="section">		
	<h2>Warning</h2>
	<p>Loud sounds can cause equipment/hearing damage with even a single, brief exposure. <b>Always turn your audio system level down to a reasonable level, before playing our sound files.</b></p>
	</div><div class="section">
<h2>Help Me Help You!</h2>
<p>Is AudioCheck free? Not for me. <span class="warning">Your support keeps this site running</span>. Any donation will be rewarded with &bull; <span class="warning">uncompressed .wav file download for every test</span> (a download arrow will appear next to each sound icon) &bull; increased durations and sample rates up to 192 kHz in the Tone Gen section &bull; and, best of all, the removal of these pesky payment buttons below &#128540;
<div style="display:inline;"><img src="/stephane.png" width=110px></div>
</p>



<div id="smart-button-container">
      <div style="text-align: left;">
        <div style="margin-bottom: 1.25rem;">
          <select id="item-options">
          	  <option value="Five" price="5.00">High Five - 5 USD</option>
			  <option value="Ten" price="10.00" selected>Perfect Ten - 10 USD</option>
			  <option value="Awesome" price="20.00">You're Awesome - 20 USD</option>
			  <option value="Best" price="30.00">You = THE BEST - 30 USD</option>
          </select>
          <select style="visibility: hidden" id="quantitySelect"></select>
        </div>
      <div id="paypal-button-container"></div>
      </div>
    </div>
    <script src="https://www.paypal.com/sdk/js?client-id=AewZfXuTDmA35htyyPijBRdlEBCF5mklutXT3Xdb-tOsxoFepFoPiol1AUlFrhpq4CdnUUEakTQSsQN0&enable-funding=venmo&currency=USD" data-sdk-integration-source="button-factory"></script>
    <script>
      function initPayPalButton() {
        var shipping = 0;
        var itemOptions = document.querySelector("#smart-button-container #item-options");
    var quantity = parseInt();
    var quantitySelect = document.querySelector("#smart-button-container #quantitySelect");
    if (!isNaN(quantity)) {
      quantitySelect.style.visibility = "visible";
    }
    var orderDescription = 'Patron Access To AudioCheck';
    if(orderDescription === '') {
      orderDescription = 'Item';
    }
    paypal.Buttons({
      style: {
        shape: 'pill',
        color: 'blue',
        layout: 'vertical',
        label: 'paypal',
        
      },
      createOrder: function(data, actions) {
        var selectedItemDescription = itemOptions.options[itemOptions.selectedIndex].value;
        var selectedItemPrice = parseFloat(itemOptions.options[itemOptions.selectedIndex].getAttribute("price"));
        var tax = (0 === 0 || false) ? 0 : (selectedItemPrice * (parseFloat(0)/100));
        if(quantitySelect.options.length > 0) {
          quantity = parseInt(quantitySelect.options[quantitySelect.selectedIndex].value);
        } else {
          quantity = 1;
        }

        tax *= quantity;
        tax = Math.round(tax * 100) / 100;
        var priceTotal = quantity * selectedItemPrice + parseFloat(shipping) + tax;
        priceTotal = Math.round(priceTotal * 100) / 100;
        var itemTotalValue = Math.round((selectedItemPrice * quantity) * 100) / 100;

        return actions.order.create({
          purchase_units: [{
            description: orderDescription,
            invoice_id: orderDescription+" #"+Date.now(),
            amount: {
              currency_code: 'USD',
              value: priceTotal,
              breakdown: {
                item_total: {
                  currency_code: 'USD',
                  value: itemTotalValue,
                },
                shipping: {
                  currency_code: 'USD',
                  value: shipping,
                },
                tax_total: {
                  currency_code: 'USD',
                  value: tax,
                }
              }
            },
            items: [{
              name: selectedItemDescription,
              unit_amount: {
                currency_code: 'USD',
                value: selectedItemPrice,
              },
              quantity: quantity
            }]
          }],
        	application_context: {
                shipping_preference: 'NO_SHIPPING',
                return_url: 'https://www.audiocheck.net/thankyou.php',
    			cancel_url: 'https://www.audiocheck.net/showMessage.php?id=2'
  			}
        });
      },
     onApprove: function(data, actions) {
        return actions.order.capture().then(function(details) {
           //alert('Transaction completed by ' + details.payer.name.given_name + '!');
           window.location.href = "https://www.audiocheck.net/thankyou.php";
        });
      },
      onError: function(err) {
        console.log(err);
        window.location.href = "https://www.audiocheck.net/showMessage.php?id=3";
      },
    }).render('#paypal-button-container');
  }
  initPayPalButton();
    </script>


<p>
If you already are a patron, please <a href="/login.php">log in</a>.
</p>
</div>	
</div>

<div id="footer">
&copy; 2009-2022 AudioCheck.net / <a href="https://stephanepigeon.com">Dr. Ir. Stéphane Pigeon</a> — For personal use only.
</div>

<script async src="//static.getclicky.com/101366498.js"></script>
<noscript><p><img alt="Clicky" width="1" height="1" src="//in.getclicky.com/101366498ns.gif" /></p></noscript>

<!--Amazon One Link
<script src="//z-na.amazon-adsystem.com/widgets/onejs?MarketPlace=US&adInstanceId=e40d75ed-036d-443d-8522-7db3d0c10cae"></script>
-->
</div>
		
	</BODY>
</HTML>