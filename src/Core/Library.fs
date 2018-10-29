namespace Arbetsformedlingen
open System
open System.IO
open FSharp.Data

module Utv=
    type Annons={
        id:string
        title:string
        url:string
        relevans:int
    }
    module Annons=
        let file_name (a:Annons) = sprintf "data/%s.json" a.id
        // /platsannonser/
        let url (a:Annons)= sprintf "http://api.arbetsformedlingen.se/af/v0/platsannonser/%s" a.id
// http://api.arbetsformedlingen.se/af/v0/platsannonser/matchning?yrkesid=80&sida=0&antalrader=1000
// Accept : application/json
// Accept-Language : sv
    type Platsannonser= JsonProvider<"./platsannonser_yrkesid_80.json">
    let platsannonser =Platsannonser.Load "./platsannonser_yrkesid_80.json"
    let matchingsdata= platsannonser.Matchningslista.Matchningdata
                        |> Array.filter (fun md->md.YrkesbenamningId = 80 ) 
                        |> Array.map (fun md->{
                                            id= md.Annonsid.JsonValue.AsString()
                                            title = md.Annonsrubrik
                                            url = md.Annonsurl
                                            relevans = md.Relevans
                                         })
                                         
    let tryPull (a:Annons)=
        let fn = Annons.file_name a
        if not <| File.Exists fn then
            let u = Annons.url a
            File.WriteAllText( fn, Http.RequestString( url= u, httpMethod="GET",
                                                       headers = [ "Accept", "application/json"; "Accept-Language", "sv"]
                                                       ))

    let print ()=
        for a in matchingsdata do
            printfn "%s : %s" a.id a.title
    let getAll ()=
        for a in matchingsdata do
            tryPull a

module Text=
  let alias =[ 
              ["developer"; "swutvecklare";  "utvecklare"; ], "_developer_"
              ["programming"; "utveckling"; "programmering"; "development" ], "_programming_"
              ["services"; "service" ], "_services_"
              ["js"; "javascript" ], "_javascript_"
              ["linux"; "_linux_miljö"], "_linux_"
              ["windows"; "_windows_miljö"; ], "_windows_"
              ["frontend"; "front-end"; "front end"], "_frontend_"
              ["backend"; "back-end"; "back end"], "_backend_"
              ["fullstack"; "full-stack"; "full stack"], "_fullstack_"
             ]
             //|> List.map (fun (alias,to')->set alias,to')
  let mapAlias (text:string)=
    let mutable res =text
    for (list,to') in alias do
      for v in list do
        res <- res.Replace(v,to')
    res

  let postfixWords = ["_developer_"; "_programming_"; "_service_"]
  let insertPostFix (text:string)=
    let pickPostfix (postfix :string)= 
      let index = text.IndexOf postfix
      if index >0 then Some (postfix,index) else None
    match postfixWords |> List.tryPick pickPostfix with
    | Some (_,index)-> text.Insert(index, "-")
    | None -> text
  let normalize = mapAlias >> insertPostFix

  let whitespaceChars = [
                  '\r'
                  '\n'
                  ' '
                  '\t'
                  '\u00A0'
                  ]
  let punctuationChars = [
                  ';'
                  ':'
                  '.'
                  '!'
                  ','
                  '?'
                  ]
  let andChars = ['&']
  let otherChars = ['/';')';'('; '•';'·';'\'';'"';'’';'-';'_';'*';'–']
  let splitChars = whitespaceChars @ punctuationChars @ andChars @ otherChars |> List.toArray
  let splitOnChars (text:string)=text.Split(splitChars, StringSplitOptions.RemoveEmptyEntries)
module ProgrammingLanguages=
  type T =HtmlProvider<"./Timeline-of-programming-languages-Wikipedia.htm">
  let t =T.Load("./Timeline-of-programming-languages-Wikipedia.htm")
  let inline getName< ^a when ^a : ( member get_Name: unit->String )>(r:^a) =
    ( ^a : ( member get_Name: unit->String ) (r) )
  let probNotRelevant = ["e";"it"]
  let rows = [ t.Tables.``1950s``.Rows |> Array.map getName
               t.Tables.``1960s``.Rows |> Array.map getName
               t.Tables.``1970s``.Rows |> Array.map getName
               t.Tables.``1980s``.Rows |> Array.map getName
               t.Tables.``1990s``.Rows |> Array.map getName
               t.Tables.``1990s``.Rows |> Array.map getName
               t.Tables.``2000s``.Rows |> Array.map getName
               t.Tables.``2010s``.Rows |> Array.map getName
             ]
             |> List.collect List.ofArray
             |> List.map (fun s->s.ToLower())
             |> List.filter (fun s-> not <| List.contains s probNotRelevant )
             |> set
             
module Annons=
  type Annons={
      id:string
      title:string
      text:string
  }
  type T=JsonProvider<"./sample_22898479.json">
  //let t = T.Load "./sample_22898479.json"
  let map (a:T.Annons) = {id=a.Annonsid; title=a.Annonsrubrik; text=a.Annonstext}
  let getAll ()=
    let files = Directory.GetFiles("./data/")
    let currentDir =Directory.GetCurrentDirectory()
    let loadAndMap (f:string)=
      let t = T.Load ( Path.Combine(currentDir, f) )
      map t.Platsannons.Annons
    let splitAndFilter (text:string)=
      Text.normalize text 
      |> Text.splitOnChars
      |> Array.toList
    let loaded = files 
                    |> Array.map loadAndMap 
                    |> Array.toList


    let onlyLangs = List.filter (fun s->Set.contains s ProgrammingLanguages.rows )
    let splitOnChars (text:string)=
        let splitChars = Text.punctuationChars@ Text.whitespaceChars |>List.toArray
        text.Split(splitChars, StringSplitOptions.RemoveEmptyEntries) |> List.ofArray
    let adAndLanguage =loaded
                       |> List.map (fun a-> a.id, splitOnChars a.title @ splitOnChars a.text 
                                                  |> List.map (fun s->s.ToLower()) 
                                                  |> onlyLangs |> List.distinct )
    let langTags = adAndLanguage
                   |> List.map (fun (id,list)-> sprintf "%s : %s" id (String.concat ", " list) )
                   |> String.concat "\n"
    File.WriteAllText ("langs.txt", langTags)
    let wordCounts= adAndLanguage
                    |> List.collect (fun (_, langs)-> langs)
                    |> List.groupBy (fun s->s.ToLower())
                    |> List.map (fun (s,l)->(s,l.Length))
                    |> List.filter(fun (_,l)->l>1)
    let txt =wordCounts
              |> List.sortByDescending (fun (_,l)->l)
              |> List.map (fun (s,l)-> sprintf "%d : %s" l s )
              |> String.concat "\n" 
    File.WriteAllText ("list-langs.txt", txt)
module Main=

    [<EntryPoint>]
    let main argv =
        //Utv.getAll()
        Annons.getAll()
        //printf "%s" ( Directory.GetCurrentDirectory() )
        0 // return an integer exit code
