# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
function load_versions(json){
    var button = document.getElementById('version-switcher-button')
    var container = document.getElementById('version-switcher-dropdown')
    var loc = window.location.href;
    var s = document.createElement('select');
    s.style = "border-radius:5px;"
    const versions = JSON.parse(json);
    for (entry of versions){
        var o = document.createElement('option');
        var optionText = '';
        if ('name' in entry){
            optionText = entry.name;
        }else{
            optionText = entry.version;
        }
        o.value = entry.url;
        if (current_version == entry.version){
            o.selected = true;
        }
        o.innerHTML = optionText;
        s.append(o);
    }
    s.addEventListener("change", ()=> {
        var current_url = new URL(window.location.href);
        var path = current_url.pathname;
        //strip version from path
        var page_path = path.substring(project_name.length+current_version.length+3);
        window.location.href = s.value + page_path;
    });
    container.append(s);
}
