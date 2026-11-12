var Se="http://www.w3.org/2000/svg",Te=new Set(["animate","animateMotion","animateTransform","circle","clipPath","defs","desc","ellipse","feBlend","feColorMatrix","feComponentTransfer","feComposite","feConvolveMatrix","feDiffuseLighting","feDisplacementMap","feDistantLight","feDropShadow","feFlood","feFuncA","feFuncB","feFuncG","feFuncR","feGaussianBlur","feImage","feMerge","feMergeNode","feMorphology","feOffset","fePointLight","feSpecularLighting","feSpotLight","feTile","feTurbulence","filter","foreignObject","g","image","line","linearGradient","marker","mask","metadata","mpath","path","pattern","polygon","polyline","radialGradient","rect","set","stop","svg","switch","symbol","text","textPath","tspan","use","view"]),we=["value","checked","selected"];function be(o,t){if(!(t==null||typeof t=="boolean")){if(Array.isArray(t)){for(let i of t)be(o,i);return}o.appendChild(typeof t=="object"?t:document.createTextNode(String(t)))}}function De(o,t){if(typeof t=="string"){o.style.cssText=t;return}if(!(!t||typeof t!="object"))for(let[i,f]of Object.entries(t))f!=null&&(i.startsWith("--")?o.style.setProperty(i,String(f)):o.style[i]=String(f))}function ye(o,t,i,f){if(t==="style"){De(o,i);return}if(t==="class"||t==="className"){i!=null&&i!==!1&&o.setAttribute("class",String(i));return}if(t.length>2&&t.startsWith("on")&&typeof i=="function"){o.addEventListener(t.slice(2).toLowerCase(),i);return}if(i!=null){if(t==="innerHTML"||t==="textContent"||!f&&!t.includes("-")&&t in o){o[t]=i;return}i!==!1&&o.setAttribute(t,i===!0?"":String(i))}}function Ie(o,t){let i=Te.has(o),f=i?document.createElementNS(Se,o):document.createElement(o);be(f,t.children);for(let[b,Q]of Object.entries(t))b==="children"||b==="ref"||we.includes(b)||ye(f,b,Q,i);for(let b of we)b in t&&ye(f,b,t[b],i);return t.ref?.(f),f}function u(o,t){return typeof o=="function"?o(t):Ie(o,t)}var je="0.9.0",xe=`
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 6px;
  background: rgba(255,255,255,0.25);
  border-radius: 9999px;
  outline: none;
  cursor: pointer;
`,_e=`
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #7c3aed;
    border: 2px solid #7c3aed;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    cursor: pointer;
    transition: box-shadow 0.15s;
  }
  input[type=range]::-webkit-slider-thumb:hover {
    box-shadow: 0 0 0 4px rgba(124,58,237,0.3);
  }
  input[type=range]::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #7c3aed;
    border: 2px solid #7c3aed;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    cursor: pointer;
    transition: box-shadow 0.15s;
  }
  input[type=range]::-moz-range-thumb:hover {
    box-shadow: 0 0 0 4px rgba(124,58,237,0.3);
  }
  input[type=range]:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  input[type=range]:disabled::-webkit-slider-thumb {
    cursor: not-allowed;
  }
  .mask-adjustment-preview input[type=number]::-webkit-outer-spin-button,
  .mask-adjustment-preview input[type=number]::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
`,T=o=>o.stopPropagation();function Ve(o,t){let i=o,{value:f,disabled:b,onChange:Q}=t,oe=f?.original_video_url||"",ke=f?.mask_video_url||"",Me=f?.adjustment??0;if(i._maskPreview){let e=i._maskPreview;if(e.wrapper&&e.wrapper.isConnected)return e.handleUpdate(t),{cleanup:e.cleanup,update:e.handleUpdate};delete i._maskPreview}let R=oe,O=ke,v=Me,_=f?.current_frame??0,k=f?.total_frames??0,V=!1,D=!1,$=!1,X=Q,Z=t.onLiveChange,W=(t.locked||[]).includes("adjustment"),c=!oe,C=30,m=document.createElement("canvas"),q=m.getContext("2d",{willReadFrequently:!0}),H=null,se="mask-preview-slider-styles";document.getElementById(se)||document.head.append(u("style",{id:se,children:_e}));let s=u("video",{style:{width:"100%",display:c?"none":"block"},preload:"auto",crossOrigin:"anonymous"}),a=u("video",{style:c?"width: 100%; display: block;":"display: none;",preload:"auto",crossOrigin:"anonymous"}),p=u("canvas",{style:{position:"absolute",top:"0",left:"0",pointerEvents:"none"}}),ee=p.getContext("2d",{willReadFrequently:!0});function B(){return c?a:s}function N(){let e=B(),n=e.clientWidth,r=e.clientHeight;n&&r&&(p.style.width=n+"px",p.style.height=r+"px")}function de(e){e=Math.max(-25,Math.min(25,e)),v=e,G.value=String(e),P.value=String(e),M(),typeof X=="function"&&X({original_video_url:R,mask_video_url:O,adjustment:e,current_frame:_,total_frames:k})}let G=u("input",{type:"range",min:"-25",max:"25",value:String(v),disabled:b||W,style:xe+"flex: 1; min-width: 0;",onPointerDown:T,onMouseDown:T,onInput:e=>{let n=parseInt(e.target.value,10);v=n,P.value=String(n),M(),typeof Z=="function"&&Z({adjustment:n})},onChange:e=>de(parseInt(e.target.value,10))}),P=u("input",{type:"number",min:"-25",max:"25",step:"1",value:String(v),disabled:b||W,style:`
        width: 52px;
        flex-shrink: 0;
        background: transparent;
        border: none;
        color: #fff;
        font-size: 12px;
        text-align: right;
        outline: none;
        -moz-appearance: textfield;
        padding: 2px 0;
      `,onPointerDown:T,onMouseDown:T,onBlur:()=>{let e=parseInt(P.value,10);isNaN(e)?P.value=String(v):de(e)},onKeyDown:e=>{e.key==="Enter"&&(e.preventDefault(),P.blur())}}),le=u("span",{style:{fontWeight:"600",color:"#fff"}}),te=u("input",{type:"range",min:"0",max:"100",value:"0",style:xe,onPointerDown:T,onMouseDown:T,onInput:e=>{let n=parseInt(e.target.value,10);_=n,A(),ue(n)}});function A(){let e=B().duration,n=k>0?k:e?Math.floor(e*C):0;le.textContent=`${_} / ${n}`,te.max=String(Math.max(0,n-1))}let I=u("span",{children:"Waiting for video inputs..."}),ne=u("div",{class:"mask-adjustment-preview nodrag nowheel",style:{display:"flex",flexDirection:"column",gap:"12px",padding:"12px",background:"#1a1a1a",borderRadius:"8px",minHeight:"400px"},children:[u("div",{style:{position:"relative",width:"100%",background:"#000",borderRadius:"4px",overflow:"hidden"},onPointerDown:T,onMouseDown:T,children:[s,a,p]}),u("div",{style:{display:"flex",flexDirection:"column",gap:"16px",padding:"12px",background:"#252525",borderRadius:"4px"},children:[u("div",{style:{display:"flex",flexDirection:"column",gap:"8px"},children:[u("span",{style:{fontSize:"12px",color:"#ccc",flexShrink:"0"},children:"Mask Adjustment"}),u("div",{style:{display:"flex",alignItems:"center",gap:"8px",width:"100%"},children:[G,P]})]}),u("div",{style:{display:"flex",flexDirection:"column",gap:"8px"},children:[u("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",fontSize:"12px",color:"#ccc"},children:[u("span",{children:"Frame"}),le]}),te]})]}),u("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",fontSize:"11px",color:"#888",padding:"8px",background:"#1f1f1f",borderRadius:"4px"},children:[I,u("span",{style:{fontSize:"10px",color:"#666",fontFamily:"monospace"},children:`v${je}`})]})]});function ue(e){let n=B();if(!n.duration||$)return;let r=k>0?k:Math.floor(n.duration*C),l=e/r*n.duration;$=!0,n.currentTime=l,!c&&a.duration&&(a.currentTime=l),c&&s.duration&&(s.currentTime=l)}function K(){if(!a.videoWidth)return!1;let e=a.videoWidth,n=a.videoHeight;(m.width!==e||m.height!==n)&&(m.width=e,m.height=n);try{return q.drawImage(a,0,0),H=q.getImageData(0,0,e,n),!0}catch{return!1}}let ie=!1;function M(){ie||(ie=!0,requestAnimationFrame(()=>{ie=!1,Le()}))}function Le(){if(!D||!H&&!K())return;let e=H;if(!c&&(!V||!s.videoWidth))return;let n=B();(p.width!==n.videoWidth||p.height!==n.videoHeight)&&(p.width=n.videoWidth,p.height=n.videoHeight),N(),ee.clearRect(0,0,p.width,p.height);try{let r=v!==0?Ce(e,v):e;if(c){if(v!==0){let l=e.width,g=e.height;(m.width!==l||m.height!==g)&&(m.width=l,m.height=g),q.putImageData(r,0,0),ee.drawImage(m,0,0,p.width,p.height)}I.textContent=`Mask: OK | ${v!==0?`Adjusted preview (${v}px)`:"No adjustment"}`}else{let l=e.width,g=e.height,x=r.data,y=new Uint8ClampedArray(l*g*4);for(let h=0;h<x.length;h+=4)x[h]>127&&(y[h]=255,y[h+1]=255,y[h+2]=0,y[h+3]=128);(m.width!==l||m.height!==g)&&(m.width=l,m.height=g),q.putImageData(new ImageData(y,l,g),0,0),ee.drawImage(m,0,0,p.width,p.height),I.textContent=`Video: OK | Mask: OK | Overlay active (adj: ${v}px)`}}catch(r){I.textContent=`Overlay error: ${r.message}`,console.error("Error rendering mask overlay:",r)}}function Ce(e,n){if(n===0)return e;let r=e.width,l=e.height,g=e.data,x=Math.abs(n),y=n>0,h=127,j=new Uint8Array(r*l);for(let d=0;d<j.length;d++)j[d]=g[d*4]>h?1:0;let L=r+1,E=new Int32Array((l+1)*L);for(let d=0;d<l;d++)for(let w=0;w<r;w++)E[(d+1)*L+(w+1)]=j[d*r+w]+E[d*L+(w+1)]+E[(d+1)*L+w]-E[d*L+w];let J=new Uint8Array(r*l);for(let d=0;d<l;d++){let w=Math.max(0,d-x),S=Math.min(l-1,d+x),Ee=S-w+1;for(let F=0;F<r;F++){let re=Math.max(0,F-x),ae=Math.min(r-1,F+x),ve=E[(S+1)*L+(ae+1)]-E[w*L+(ae+1)]-E[(S+1)*L+re]+E[w*L+re];y?J[d*r+F]=ve>0?1:0:J[d*r+F]=ve===Ee*(ae-re+1)?1:0}}let z=new Uint8ClampedArray(g.length);for(let d=0;d<J.length;d++){let w=J[d]?255:0,S=d*4;z[S]=z[S+1]=z[S+2]=w,z[S+3]=255}return new ImageData(z,r,l)}function U(){let e=[];c||(R?V?e.push("Video: OK"):e.push("Loading video..."):e.push("No original video")),O?D?e.push("Mask: OK"):e.push("Loading mask..."):e.push("No mask video"),c&&e.push("Mask-only preview"),I.textContent=e.join(" | ")}function Y(e){return e?e.split("?")[0]:""}function ce(e){e&&Y(s.src)!==Y(e)&&(V=!1,U(),s.src=e)}function fe(e){e&&Y(a.src)!==Y(e)&&(D=!1,H=null,U(),a.src=e)}function pe(){_>0&&(te.value=String(_),ue(_))}function me(e,n){if(!e.requestVideoFrameCallback)return;let r=null;e.requestVideoFrameCallback((l,g)=>{r=g.mediaTime,e.requestVideoFrameCallback((x,y)=>{let h=y.mediaTime-r;if(h>0){let j=Math.round(1/h);j>0&&j<240&&n(j)}})})}s.addEventListener("loadedmetadata",()=>{c||(p.width=s.videoWidth,p.height=s.videoHeight,N(),me(s,e=>{C=e,k=Math.floor(s.duration*C),A()}),k=Math.floor(s.duration*C),A()),V=!0,U(),D&&(pe(),M())}),s.addEventListener("loadeddata",()=>{!c&&D&&H&&M()}),s.addEventListener("resize",N),a.addEventListener("resize",()=>{c&&N()}),a.addEventListener("loadedmetadata",()=>{D=!0,c&&(p.width=a.videoWidth,p.height=a.videoHeight,N(),me(a,e=>{C=e,k=Math.floor(a.duration*C),A()}),k=Math.floor(a.duration*C),A()),U(),a.requestVideoFrameCallback&&a.requestVideoFrameCallback(()=>{K()&&M()}),(c||V)&&pe()}),a.addEventListener("loadeddata",()=>{D&&(c||V)&&(H||K(),M())}),a.addEventListener("error",e=>{I.textContent=`Mask video error: ${a.error?.message||"unknown"}`,console.error("Mask video error:",e,a.error)}),s.addEventListener("error",e=>{I.textContent=`Video error: ${s.error?.message||"unknown"}`,console.error("Original video error:",e,s.error)}),s.addEventListener("seeked",()=>{$=!1,c||M()}),a.addEventListener("seeked",()=>{$=!1,K(),M()}),ce(R),fe(O),A(),i.appendChild(ne);function Pe(e,n,r,l,g,x,y){l&&(X=l),Z=g,W=(x||[]).includes("adjustment"),G.disabled=y||W,P.disabled=y||W;let h=!e;h!==c&&(c=h,s.style.display=c?"none":"block",a.style.cssText=c?"width: 100%; display: block;":"display: none;"),e&&e!==R&&(R=e,ce(e)),n&&n!==O&&(O=n,fe(n)),r!==v&&(v=r,G.value=String(r),P.value=String(r),M())}function he(){s.pause(),a.pause(),s.removeAttribute("src"),a.removeAttribute("src"),s.load(),a.load(),s.remove(),a.remove(),ne.remove(),delete i._maskPreview}function ge(e){let n=e.value;Pe(n?.original_video_url||"",n?.mask_video_url||"",n?.adjustment??0,e.onChange,e.onLiveChange,e.locked,e.disabled)}return i._maskPreview={handleUpdate:ge,cleanup:he,wrapper:ne},{cleanup:he,update:ge}}export{Ve as default};
